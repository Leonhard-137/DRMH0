"""Exercise the real FastPairLikelihood resume without advancing any fit."""
from pathlib import Path
import hashlib
import json
import pickle
import sys

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from mica_slowgp.cli import read_data
from mica_slowgp.fast import FastPairLikelihood
from mica_slowgp.nested import NestedBudget
import mica_slowgp.stratified as stratified

folder = Path('/home/leonhard/sci001/output/mica_slowgp/stability_v3_stratified/dynesty/shared_shared/budget_400/seed_71231')
config = json.loads((folder / 'sampler_config.json').read_text())
child_config = json.loads((folder / 'strata/stratum_00/sampler_config.json').read_text())
inputs = [config['run_input'][name] for name in ('driver', 'response')]
callback = FastPairLikelihood(read_data(*inputs), mode=config['mode'])
parent_payload = pickle.dumps(callback, protocol=pickle.HIGHEST_PROTOCOL)
child_payload = pickle.dumps(pickle.loads(parent_payload), protocol=pickle.HIGHEST_PROTOCOL)
digest = lambda value: hashlib.sha256(value).hexdigest()
assert digest(parent_payload) == config['likelihood_pickle_sha256']
assert digest(child_payload) == child_config['likelihood_pickle_sha256']
assert digest(parent_payload) != digest(child_payload)

before = {str(p): (p.stat().st_mtime_ns, digest(p.read_bytes()))
          for p in folder.rglob('*') if p.is_file()}

class ExpectedNextChild(Exception):
    pass

calls = []
def intercept_next_child(callback, output_dir, **kwargs):
    calls.append(dict(output=str(output_dir), seed=kwargs['seed'],
                      center_interval=kwargs['center_interval']))
    assert Path(output_dir).name == 'stratum_01'
    assert kwargs['seed'] == config['child_seeds'][1]
    assert kwargs['center_interval'] == config['center_strata'][1:3]
    raise ExpectedNextChild

stratified.run_nested = intercept_next_child
try:
    stratified.run_stratified(callback, folder, mode=config['mode'], priors=config['priors'],
        budget=NestedBudget(**config['budget']), seed=config['seed'],
        center_strata=config['center_strata'], resume=True,
        input_files=inputs, run_input=config['run_input'])
except ExpectedNextChild:
    pass
else:
    raise AssertionError('Expected validated child reuse followed by child_01')
assert len(calls) == 1
after = {str(p): (p.stat().st_mtime_ns, digest(p.read_bytes()))
         for p in folder.rglob('*') if p.is_file()}
assert before == after
report = dict(passed=True, parent_identity_preserved=True,
    child_identity_matches_actual_clone=True, completed_child_reused=True,
    next_child=calls[0], original_files_and_mtimes_unchanged=True,
    scientific_likelihood_calls=0, new_scientific_fits=0,
    parent_pickle_sha256=digest(parent_payload), child_pickle_sha256=digest(child_payload))
output = Path(__file__).with_name('check_report.json')
output.write_text(json.dumps(report, indent=2))
print(json.dumps(report))
