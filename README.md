
# Usage

```python
bash test/run_test.sh test/test_online_recorder.py
```

Offline trace processing
```python
python3 offline/collect.py --torch_trace logs
```

Use `--combine` to combine the processed traces and the original torch traces, such that the processed traces can serve as annotations for the original torch traces. Refer to [Issue #1](https://github.com/joapolarbear/dpro-3d/issues/1#issue-2014346849)