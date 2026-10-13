"""Replay frozen alerts on native contiguous time segments; no file reads."""
import numpy as np
from congestion_alert_pilot import event_stats, point_stats
from congestion_risk_pilot import online_alarm
from congestion_objective_diagnostic import trace_events

COLUMNS = ['exposure', 'events', 'warned', 'detected', 'false', 'duplicate', 'late',
           'censored_events', 'censored_alerts', 'unknown_alerts', 'warning5', 'warning10', 'warning15']


def layout(part):
    origins = part['origins']; times = part['timestamps'][origins]
    if not len(origins): raise ValueError('No replay origins')
    breaks = np.flatnonzero((np.diff(times) != np.timedelta64(5,'m')) | (np.diff(origins) != 1))+1
    edges = np.r_[0, breaks, len(origins)]
    d0 = part['timestamps'][0].astype('datetime64[D]')
    end = part['timestamps'][-1].astype('datetime64[D]')+np.timedelta64(1,'D')
    dates = np.arange(d0,end,dtype='datetime64[D]')
    day_ids = (times.astype('datetime64[D]')-d0).astype(int)
    return list(zip(edges[:-1],edges[1:])), day_ids, dates


def alarm(scores, threshold, confirmation, part):
    if confirmation not in (1,2): raise ValueError('Unsupported confirmation')
    result = online_alarm(scores,threshold,confirmation)
    if confirmation == 2:
        for lo,_ in layout(part)[0]: result[lo] = False
    return result


def replay(part, activated, details=False, scenario=None):
    """Keep natural three-reading eligibility; reset only at clock discontinuity."""
    segments,day_ids,dates = layout(part)
    daily = np.zeros((len(dates),13),np.float64)
    event_records=[]
    for lo,hi in segments:
        truth,eligible = part['truth'][lo:hi],part['eligible'][lo:hi]
        bits = activated[lo:hi]; days=day_ids[lo:hi]
        stats,core,_ = event_stats(truth,eligible,bits,days)
        daily[:len(core),:10] += core
        # Legacy trace needs at least one complete event; the core scorer does not.
        if details and stats['events'] > 0:
            reconstructed,ref,records,_ = trace_events(truth,eligible,bits,days,part['origins'][lo:hi],part['timestamps'])
            assert reconstructed==stats and np.array_equal(ref,core)
            for rec in records:
                rec['segment_first_origin']=int(part['origins'][lo])
                rec['segment_last_origin']=int(part['origins'][hi-1])
                if rec['status']=='warned':
                    day = int((part['timestamps'][rec['event_start_origin']].astype('datetime64[D]')-dates[0])/np.timedelta64(1,'D'))
                    daily[day,9+rec['lead_minutes']//5] += 1
            event_records.extend(records)
    total=daily.sum(axis=0); count=int(total[1]); exposure=float(total[0])
    leads=[r['lead_minutes'] for r in event_records if r['status']=='warned']
    delays=[r['late_minutes'] for r in event_records if r['status']=='late']
    row={k:float(v) for k,v in zip(COLUMNS[:10],total[:10])}
    row.update({'warning_recall':float(total[2]/count) if count else None,
                'detection_recall':float(total[3]/count) if count else None,
                'false_per_valid_station_day':float(total[4]/exposure) if exposure else None,
                'missed_events':int(total[1]-total[3])})
    if details:
        assert np.array_equal(daily[:,10:13].sum(axis=1),daily[:,2])
        row.update({'warning_recall_at_least10':float(total[11:13].sum()/count) if count else None,
                    'warning_lead_counts':{str(k):int(total[9+k//5]) for k in (5,10,15)},
                    'warning_lead_mean_minutes':float(np.mean(leads)) if leads else None,
                    'warning_lead_median_minutes':float(np.median(leads)) if leads else None,
                    'late_detection_mean_minutes':float(np.mean(delays)) if delays else None,
                    'point_all':point_stats(part['truth'],part['eligible'],activated)})
        if scenario is not None:
            fault=part[f'{scenario}_fault']
            row['point_fault_nodes']=point_stats(part['truth'],part['eligible']&fault,activated)
            row['point_other_nodes']=point_stats(part['truth'],part['eligible']&~fault,activated)
    return row,daily,event_records


def select_key(rows, confirmation, index):
    warning=[r['warning_recall'] for r in rows]
    if any(v is None for v in warning): raise ValueError('No complete calibration events')
    return (-float(np.mean(warning)),-float(np.min(warning)),
            float(np.mean([r['false_per_valid_station_day'] for r in rows])),confirmation,index)
