import argparse
import csv
import os
import re
import subprocess
import sys
from datetime import datetime


def make_log_dir(base_script_dir, experiment_name='humanoid_n1', run_name='ppo_n1'):
    root = os.path.join(base_script_dir, 'logs')
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = os.path.join(root, experiment_name, f'{run_name}_{timestamp}')
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def sanitize_key(s: str) -> str:
    s = s.strip().replace(' ', '_').replace('/', '__')
    s = s.replace(':', '')
    return s


ITER_RE = re.compile(r'Learning iteration\s+(\d+)/(\d+)')
KV_RE = re.compile(r'^\s*([^:]+):\s+(.+?)\s*$')
CHECKPOINT_RE = re.compile(r'Logs/checkpoints saved under:\s*(.+)$')
TIME_RE = re.compile(r'(?:(\d+):)?(\d+):(\d+)$')


def parse_value(text: str):
    t = text.strip()
    if t in {'True', 'False'}:
        return t == 'True'
    m = TIME_RE.match(t)
    if m:
        h = int(m.group(1) or 0)
        mi = int(m.group(2))
        s = int(m.group(3))
        return h * 3600 + mi * 60 + s
    try:
        if any(c in t for c in '.eE'):
            return float(t)
        return int(t)
    except Exception:
        return t


def maybe_write_row(writer, fieldnames, current, csv_path):
    if not current or 'iteration' not in current:
        return writer, fieldnames
    row = dict(current)
    if writer is None:
        fieldnames = list(row.keys())
        f = open(csv_path, 'w', newline='')
        writer = (f, csv.DictWriter(f, fieldnames=fieldnames))
        writer[1].writeheader()
    else:
        f, dw = writer
        for k in fieldnames:
            row.setdefault(k, None)
        extra = [k for k in row.keys() if k not in fieldnames]
        if extra:
            for k in extra:
                fieldnames.append(k)
            f.close()
            # rewrite whole CSV with expanded schema is annoying; easier: fail loudly
            raise RuntimeError(f'New columns appeared mid-run: {extra}. Delete CSV and rerun.')
    writer[1].writerow({k: row.get(k, None) for k in fieldnames})
    writer[0].flush()
    return writer, fieldnames



def main():
    parser = argparse.ArgumentParser(description='Run working trainer and mirror stdout metrics to CSV.')
    parser.add_argument('--trainer', type=str, default='/home/neramaswamy/humanoid_n1_task/source/NRamaswamy_CS791_Project1/humanoid_n1_task/train_n1_stand.py')
    parser.add_argument('--csv_name', type=str, default='training_metrics.csv')
    parser.add_argument('--experiment_name', type=str, default='humanoid_n1')
    parser.add_argument('--run_name', type=str, default='ppo_n1')
    args, remainder = parser.parse_known_args()

    trainer = os.path.abspath(args.trainer)
    base_dir = os.path.dirname(trainer)
    log_dir = make_log_dir(base_dir, args.experiment_name, args.run_name)
    csv_path = os.path.join(log_dir, args.csv_name)

    cmd = [sys.executable, trainer, '--log_root', os.path.join(base_dir, 'logs'), '--experiment_name', args.experiment_name, '--run_name', args.run_name]
    cmd.extend(remainder)

    print(f'[CSV-WRAPPER] trainer   : {trainer}')
    print(f'[CSV-WRAPPER] log_dir   : {log_dir}')
    print(f'[CSV-WRAPPER] csv_path  : {csv_path}')
    #print(f'[CSV-WRAPPER] command   : {' '.join(cmd)}')

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True,
    )

    current = {}
    writer = None
    fieldnames = None
    last_checkpoint_dir = None

    try:
        for raw_line in proc.stdout:
            line = raw_line.rstrip('\n')
            print(line, flush=True)

            m = ITER_RE.search(line)
            if m:
                writer, fieldnames = maybe_write_row(writer, fieldnames, current, csv_path)
                current = {
                    'iteration': int(m.group(1)),
                    'total_iterations': int(m.group(2)),
                }
                continue

            m = CHECKPOINT_RE.search(line)
            if m:
                last_checkpoint_dir = m.group(1).strip()
                current['checkpoint_dir'] = last_checkpoint_dir
                continue

            m = KV_RE.match(line)
            if m and 'iteration' in current:
                key = sanitize_key(m.group(1))
                value = parse_value(m.group(2))
                current[key] = value
                continue

        rc = proc.wait()
        writer, fieldnames = maybe_write_row(writer, fieldnames, current, csv_path)

        if writer is not None:
            writer[0].close()

        print(f'[CSV-WRAPPER] process return code: {rc}')
        print(f'[CSV-WRAPPER] csv saved at        : {csv_path}')
        if last_checkpoint_dir:
            print(f'[CSV-WRAPPER] trainer log dir     : {last_checkpoint_dir}')

        if rc != 0:
            sys.exit(rc)
    finally:
        try:
            if proc.poll() is None:
                proc.terminate()
        except Exception:
            pass


if __name__ == '__main__':
    main()
