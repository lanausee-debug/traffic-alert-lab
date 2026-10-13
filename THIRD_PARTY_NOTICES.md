# Attribution and dependencies

PEMS-BAY access/provenance refers to the DCRNN authors: Yaguang Li, Rose Yu, Cyrus Shahabi and Yan Liu, Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting (ICLR 2018), https://arxiv.org/abs/1707.01926 . Upstream repository: https://github.com/liyaguang/DCRNN . Its software LICENSE declares MIT, copyright 2017 Yaguang Li: https://github.com/liyaguang/DCRNN/blob/master/LICENSE . This package does not vendor that repository's implementation or distribute its dataset.

PyTorch, NumPy, h5py and Matplotlib are installed separately and retain their own licenses. The graph TCN and risk MLP use established graph aggregation, temporal convolution, gating and classification ideas. They are not claimed as new architectures or exact reproductions of named external models.

The CI configuration uses official GitHub actions/checkout and actions/setup-python. Sources inspected 2026-10-13: https://github.com/actions/checkout and https://github.com/actions/setup-python .
