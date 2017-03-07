
import os

models_dir = os.path.dirname(os.path.realpath(__file__)) + '/../../models'

DEFAULT_SETTINGS = {
    "logging": {
        "display_iter": 50,
        "save_iter": 10000
    },
    "solver": {
        "opt": "RMS",
        "use_jitter": False,
        "rnd_seed": 1,
        "epsilon": 0.00001,
        "learning_rate": 0.001,
        "learning_rate_step": 33000,
        "hungarian_iou": 0.25,
        "weights": "",
        "head_weights": [1.0, 0.1]
    },
    "train_test_ratio": 0.90,
    "num_steps": 1000,
    "use_lstm": False,
    "use_rezoom": True,
    "biggest_box_px": 10000,
    "rezoom_change_loss": "center",
    "rezoom_w_coords": [-0.25, 0.25],
    "rezoom_h_coords": [-0.25, 0.25],
    "reregress": True,
    "focus_size": 1.8,
    "early_feat_channels": 256,
    "later_feat_channels": 832,
    "avg_pool_size": 5,
    "slim_top_lname": "Mixed_5b",
    "slim_attention_lname": "Mixed_3b",
    "slim_basename": "InceptionV1",
    "slim_ckpt": models_dir + '/inception_v1/state/model.ckpt',
    "num_lstm_layers": 2,
    "image_width": 640,
    "image_height": 480,
    "grid_height": 15,
    "grid_width": 20,
    "batch_size": 1,
    "region_size": 32,
    "clip_norm": 1.0,
    "lstm_size": 500,
    "deconv": False,
    "num_classes": 2,
    "rnn_len": 1
}
