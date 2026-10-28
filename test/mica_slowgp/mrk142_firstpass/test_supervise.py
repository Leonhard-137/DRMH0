#!/usr/bin/env python3
"""Lightweight tests: fake workers only, never a scientific likelihood call."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np

import prep_run
import supervise

HERE = Path(__file__).resolve().parent
WORKER = '''import json,pathlib,signal,sys,time
p=pathlib.Path(sys.argv[1]); kind=sys.argv[2]
(p/'started').write_text(str(time.monotonic()))
(p/'dynesty.checkpoint').write_bytes(b'checkpoint retained exactly')
if kind=='fail': raise SystemExit(7)
if kind=='ignore_int': signal.signal(signal.SIGINT,signal.SIG_IGN)
if kind in ('wait','ignore_int'): time.sleep(30)
else:
 time.sleep(.2)
 (p/'fake_complete').write_text(str(time.monotonic()))
'''


class FirstPassTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='_test_', dir=HERE)
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self, kinds, deadline=5):
        worker = self.root / 'worker.py'
        worker.write_text(WORKER)
        jobs = [dict(id=f'job{i}', band='U', role='repeat' if i else 'primary', seed=100+i,
                     out_dir=str(self.root/f'job{i}'), command=[sys.executable,'-B',str(worker),str(self.root/f'job{i}'),kind])
                for i,kind in enumerate(kinds)]
        settings = dict(source=str(prep_run.SOURCE), python=sys.executable, cwd=str(self.root), environment={},
                        max_parallel=6, global_fit_deadline_seconds=deadline, interrupt_grace_seconds=.3,
                        terminate_grace_seconds=.3, poll_seconds=.02, monitor_interval_seconds=.1)
        manifest = dict(root=str(self.root),jobs=jobs,run_settings=settings,postprocess_command=[sys.executable,'-B','-c',
            "from pathlib import Path; Path('postprocessed').write_text('once')"])
        path = self.root/'manifest.json'
        path.write_text(json.dumps(manifest))
        return path, manifest

    @staticmethod
    def fake_validator(job, manifest):
        if not (Path(job['out_dir'])/'fake_complete').exists():
            raise ValueError('Missing fixture completion')
        return dict(strata_complete=3)

    def test_normalization_and_physical_gain(self):
        w = np.array([[1.,4.,.2],[2.,8.,.4],[3.,6.,.3]])
        b = np.array([[1.,10.,.5],[2.,20.,1.],[3.,15.,.7]])
        wn, sw = prep_run.normalize(w)
        bn, sb = prep_run.normalize(b)
        self.assertEqual(sw,6.)
        self.assertEqual(sb,15.)
        np.testing.assert_array_equal(wn[:,0],w[:,0])
        np.testing.assert_allclose(wn[:,1:]*sw,w[:,1:])
        np.testing.assert_allclose(bn[:,1:]*sb,b[:,1:])
        self.assertAlmostEqual(prep_run.physical_gains(.4,sw,sb),1.)
        np.testing.assert_allclose(sb*(.4*wn[:,1]),prep_run.physical_gains(.4,sw,sb)*w[:,1])

    def test_fixed_preparation_no_launch(self):
        project=self.root/'project'
        for band in prep_run.BANDS:
            p=project/'Mrk142/prepared'/f'Mrk142_{band}.txt'
            p.parent.mkdir(parents=True,exist_ok=True)
            np.savetxt(p,[[1,2,.1],[2,3,.2],[3,4,.2]])
            if band!='W2':
                prior=project/'Mrk142/runs/mica/gaussian/free_raw/2comp'/f'run_W2_to_{band}_2comp_gaussian/data/para_names_line.txt_2'
                prior.parent.mkdir(parents=True)
                prior.write_text('6 0-th_component_sigma LOG -2 3 0 0\n9 1-th_component_sigma LOG -2 3 0 0\n')
        out=self.root/'prepared'
        result=prep_run.prepare(out,project=project)
        self.assertEqual(len(result['jobs']),6)
        self.assertEqual([j['seed'] for j in result['jobs']],[14201,14202,14203,14204,14205,14213])
        self.assertEqual(result['run_settings']['mode'],'shared')
        self.assertEqual(result['run_settings']['max_parallel'],6)
        self.assertEqual(result['run_settings']['center_strata'],[0,5,20,100])
        self.assertEqual(result['run_settings']['global_fit_deadline_seconds'],21600)
        self.assertEqual(result['normalization']['U']['scale'],3)
        self.assertFalse((out/'supervisor.started.json').exists())
        self.assertEqual(list((out/'runs').iterdir()),[])
        p=json.loads(Path(result['jobs'][2]['priors']).read_text())
        np.testing.assert_allclose(p['w0']['bounds'],np.exp([-2,3]))
        with self.assertRaises(FileExistsError): prep_run.prepare(out,project=project)

    def test_six_concurrent_workers_and_once_only(self):
        path,manifest=self.fixture(['complete']*6)
        state=supervise.supervise(path,self.fake_validator,start_monitor=False)
        self.assertEqual(state['state'],'finished')
        self.assertEqual(state['complete_jobs'],6)
        starts=[float((Path(j['out_dir'])/'started').read_text()) for j in manifest['jobs']]
        ends=[float((Path(j['out_dir'])/'fake_complete').read_text()) for j in manifest['jobs']]
        self.assertLess(max(starts),min(ends))
        self.assertEqual((self.root/'postprocessed').read_text(),'once')
        with self.assertRaises(FileExistsError): supervise.supervise(path,self.fake_validator,start_monitor=False)

    def test_one_failure_does_not_stop_other_jobs(self):
        path,manifest=self.fixture(['fail','complete'])
        state=supervise.supervise(path,self.fake_validator,start_monitor=False)
        self.assertEqual([j['state'] for j in state['jobs']],['failed','complete'])
        self.assertEqual(state['jobs'][0]['returncode'],7)
        self.assertEqual(state['complete_jobs'],1)

    def test_timeout_preserves_checkpoints_and_only_owns_children(self):
        path,manifest=self.fixture(['wait','ignore_int'],deadline=.35)
        unrelated=subprocess.Popen([sys.executable,'-B','-c','import time; time.sleep(30)'],start_new_session=True)
        try:
            state=supervise.supervise(path,self.fake_validator,start_monitor=False)
            self.assertEqual(state['state'],'deadline')
            self.assertEqual(state['complete_jobs'],0)
            self.assertEqual(state['last_stop_signal'],'SIGTERM')
            for job in manifest['jobs']:
                self.assertEqual((Path(job['out_dir'])/'dynesty.checkpoint').read_bytes(),b'checkpoint retained exactly')
            self.assertIsNone(unrelated.poll())
            self.assertTrue((self.root/'postprocessed').exists())
        finally:
            os.killpg(unrelated.pid,signal.SIGTERM)
            unrelated.wait(timeout=5)

    def test_completion_rejects_missing_conditional_layer(self):
        folder=self.root/'parent'
        folder.mkdir()
        priors=self.root/'priors.json'
        priors.write_text('{}')
        config=dict(mode='shared',seed=1,center_strata=[0.,5.,20.,100.],child_seeds=[11,12,13],priors={},budget={})
        (folder/'sampler_config.json').write_text(json.dumps(config))
        (folder/'completed.json').write_text('{}')
        job=dict(out_dir=str(folder),seed=1,priors=str(priors))
        manifest=dict(run_settings=dict(source=str(prep_run.SOURCE),center_strata=config['center_strata'],budget={}))
        with self.assertRaisesRegex(ValueError,'Stratum 0 is incomplete'):
            supervise.validate_complete(job,manifest)

    def test_monitor_writes_final_snapshot(self):
        supervise.atomic_json(self.root/'status.json',dict(state='finished',pid=os.getpid(),jobs=[]))
        self.assertEqual(supervise.monitor(self.root,.01),0)
        self.assertTrue((self.root/'monitor/latest.json').is_file())
        self.assertEqual(len((self.root/'monitor/progress.jsonl').read_text().splitlines()),1)


if __name__=='__main__':
    unittest.main(verbosity=2)
