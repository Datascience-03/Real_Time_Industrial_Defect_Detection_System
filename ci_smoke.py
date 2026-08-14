import os
import sys

def check_paths():
    required = [
        'runs/detect/train/weights/best.onnx',
        'dataset/test/images',
        'realtime_inference.py',
        'run_evaluation.py'
    ]
    missing = []
    for p in required:
        if not os.path.exists(p):
            missing.append(p)
    # Check for checkpoints (best.pt or last.pt) as optional but important
    ckpt_dir = 'runs/detect/train/weights'
    ckpt_found = False
    if os.path.isdir(ckpt_dir):
        for f in os.listdir(ckpt_dir):
            if f.endswith('.pt'):
                ckpt_found = True
                break

    if missing or not ckpt_found:
        print('CI_SMOKE: Missing files/folders or checkpoints:')
        for m in missing:
            print(' -', m)
        if not ckpt_found:
            print(' - No .pt checkpoint found in runs/detect/train/weights')
        # don't fail CI for missing optional artifacts, just warn
        return 1
    print('CI_SMOKE: All required files present')
    return 0

if __name__ == '__main__':
    code = check_paths()
    sys.exit(code)
