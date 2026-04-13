Testing models for a continuous stream of data. The data is gym session data with marked exercise/ non-exercise periods. The goal is to be able to analyze how well a model would act in a real scenario - how well it detects and keeps to a state.

This measures:
- Temporal stability: does the model flicker?
- Transition detection: how well and how quickly does the model detect start/end of exercise?

Visualization:
- Plot gym session showing
    - Raw data (IMU)
    - Ground truth exercise periods
    - Model predictions

Metrics:
- Detection delay: time between actual start/end and model detection
- Stability score: count of label switches per minute
- Segment accuracy: accuracy of predicted segments compared to ground truth segments
- How often false positives/negatives occur in non-exercise/exercise periods
    - does it hallucinate? does it miss real exercise?
This approach allows us to evaluate models in a way that reflects real-world usage, where data is continuous and timely detection is crucial.

Models:
- CNN
- CNN with simple state machine:
    - 2-3 consecutive exercise predictions needed to switch to exercise state
    - 2-3 consecutive non-exercise predictions needed to switch to non-exercise state
    - or last 5 predictions majority vote: based on avg probablities or on binary labels


Possible future models:
- Movement energy?
- Probabilistic models:
    - HMM, DTW(?)
- Classical ML:
    - RF, SVM, DT
- RNN/LSTM for temporal 