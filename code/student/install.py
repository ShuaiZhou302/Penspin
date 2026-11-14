from pathlib import Path
import shutil
src=Path('<PROJECT_ROOT>/penspin-16831-protocol-audit-20261012/student')
dst=Path('<PROJECT_ROOT>/penspin-16831-dagger-scratch-20261012')
dst.mkdir(exist_ok=True)
for name in ('penspin','configs'):
    if not (dst/name).exists():
        shutil.copytree(src/name,dst/name,ignore=shutil.ignore_patterns('__pycache__'))
for name in ('assets','cache','.conda','.git'):
    if not (dst/name).exists():
        (dst/name).symlink_to((src/name).resolve(),target_is_directory=True)
for name in ('runs','outputs'):
    (dst/name).mkdir(exist_ok=True)
shutil.copy2(src/'train.py',dst/'train.py')
shutil.copy2(src/'protocol_student_diagnostic.py',dst/'protocol_student_diagnostic.py')
p=dst/'penspin/algo/ppo/demon.py'
s=p.read_text()
if 'from dagger_train import run' not in s:
    s=s.replace('    def train(self):\n','    def train(self):\n        if os.environ.get("PENSPIN_DAGGER_CONTINUE") == "1":\n            from dagger_train import run\n            return run(self)\n',1)
    p.write_text(s)
print(dst)

p=dst/'train.py'
s=p.read_text()
needle='        agent.restore_train(config.train.load_path)'
assert needle in s
s=s.replace(needle, '        assert not config.train.load_path, "scratch job must not load a student checkpoint"\n'+needle, 1)
p.write_text(s)
