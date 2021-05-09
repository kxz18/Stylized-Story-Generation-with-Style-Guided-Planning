# Stylized Story Generation with Style-Guided Planning

## Training
We provide a shell script that can train and evaluate by a single command.

Before training, the data (`ROCStories_train.csv`, `ROCStories_dev.csv`, `ROCStories_test.csv`) should be copy to `data/`

Go to the root dir. If you can train four models at the same time,  try this:

```
bash run.sh parallel
```

If you computing resource is limited, please try this instead:

```
bash run.sh serial
```

After running, in each model's directory, you can find log about training and test. Besides, in `analyze/bleu`, there are reports about BLEU and in `analyze/LSC_SSC`, there are reports about LSC score and SSC score.
