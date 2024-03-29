import os
import pickle
from math import log
import certifi
from ._ncrf import fit_ncrf

from eelbrain import Categorial, concatenate
from eelbrain.testing import assert_dataobj_equal
import pytest


os.environ.setdefault('SSL_CERT_FILE', certifi.where())

# web url to fetch the file
url = "https://ece.umd.edu/~proloy/.datasets/%s.pickled"

names = ('meg', 'stim', 'fwd_sol', 'emptyroom')

# manage local storage
dirname = os.path.realpath(os.path.join(__file__, '..', '..', "ncrf_data"))
if os.path.isdir(dirname) is False:
    os.mkdir(dirname)


def load(name):
    if name in names:
        fname = os.path.join(dirname, f"{name}.pickled")
        if not os.path.isfile(fname):
            _fetch_file(url % name, fname)
        else:
            print(f"{name}.pickled already downloaded.")
    else:
        raise ValueError(f"{name}: not found")
    with open(fname, 'rb') as f:
        v = pickle.load(f)
    return v


def _fetch_file(url, fname):
    import shutil
    import tempfile
    import urllib.request

    with urllib.request.urlopen(url) as response:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            shutil.copyfileobj(response, tmp_file)
    shutil.move(tmp_file.name, fname)
    return


def test_ncrf():
    meg = load('meg').sub(time=(0, 5))
    stim = load('stim').sub(time=(0, 5))
    fwd = load('fwd_sol')
    emptyroom = load('emptyroom')

    # 1 stimulus
    model = fit_ncrf(meg, stim, fwd, emptyroom, tstop=0.2, normalize='l1', mu=0.0019444, n_iter=3, n_iterc=3,
                     n_iterf=10, do_post_normalization=False)
    # check residual and explained var
    assert model.explained_var == pytest.approx(0.00641890144769941, rel=0.001)
    assert model.voxelwise_explained_variance.sum() == pytest.approx(0.08261162457414245, rel=0.001)
    assert model.residual == pytest.approx(178.512, 0.001)
    # check scaling
    stim_baseline = stim.mean()
    assert model._stim_baseline[0] == stim_baseline
    assert model._stim_scaling[0] == (stim - stim_baseline).abs().mean()
    assert model.h.norm('time').norm('source').norm('space') == pytest.approx(6.601677e-10, rel=0.001)

    # test persistence
    model_2 = pickle.loads(pickle.dumps(model, pickle.HIGHEST_PROTOCOL))
    assert_dataobj_equal(model_2.h, model.h)
    assert_dataobj_equal(model_2.h_scaled, model.h_scaled)
    assert model_2.residual == model.residual
    assert model_2.gaussian_fwhm == model.gaussian_fwhm

    # test gaussian fwhm
    model = fit_ncrf(meg, stim, fwd, emptyroom, tstop=0.2, normalize='l1', mu=0.0019444, n_iter=1, n_iterc=1,
                     n_iterf=1, gaussian_fwhm=50.0)
    assert model.gaussian_fwhm == 50.0

    # 2 stimuli, one of them 2-d, normalize='l2'
    diff = stim.diff('time')
    stim2 = concatenate([diff.clip(0), diff.clip(max=0)], Categorial('rep', ['on', 'off']))
    model = fit_ncrf(meg, [stim, stim2], fwd, emptyroom, tstop=[0.2, 0.2], normalize='l2', mu=0.0019444, n_iter=3,
                     n_iterc=3, n_iterf=10, do_post_normalization=False)
    # check scaling
    assert model._stim_baseline[0] == stim.mean()
    assert model._stim_scaling[0] == stim.std()
    assert model.h[0].norm('time').norm('source').norm('space') == pytest.approx(7.0088e-10, rel=0.001)

    # cross-validation
    model = fit_ncrf(meg, stim, fwd, emptyroom, tstop=0.2, normalize='l1', mu='auto', n_iter=1, n_iterc=2, n_iterf=2,
                     n_workers=1, do_post_normalization=False)
    assert model.mu == pytest.approx(0.0203, 0.001)
    model.cv_info()
