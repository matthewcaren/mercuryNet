# from inference import load_model, infer_vid
import torch
from shutil import copy
from torchvision import transforms
from PIL import Image
from pathlib import Path
from glob import glob
import numpy as np
from tqdm import tqdm
import sys, cv2, os, pickle, argparse, subprocess

from model.model import MercuryNet
from hparams import hparams as hps
from utils.util import mode, to_var, to_arr

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.manual_seed(0)


class Generator(object):
    def __init__(self, model):
        super(Generator, self).__init__()

        self.synthesizer = model.eval()

    def read_window(self, window_fnames):
        window = []
        for fname in window_fnames:
            img = cv2.imread(fname)
            height, width, channels = img.shape
            path = Path(fname)
            fparent = str(path.parent.parent.parent)
            ############ transform
            if fparent[-3:] == "new":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                crop = transforms.CenterCrop((height // 2, width // 2))
                img = crop(img)
                img = np.array(img)
                # Convert RGB to BGR
                img = img[:, :, ::-1].copy()
            if img is None:
                raise FileNotFoundError(
                    "Frames maybe missing in {}."
                    " Delete the video to stop this exception!".format(sample["folder"])
                )

            img = cv2.resize(img, (hps.img_size, hps.img_size))
            window.append(img)

        images = np.asarray(window) / 255.0  # T x H x W x 3
        return images

    def vc(self, sample, outfile):
        # hp = sif.hparams
        images = sample["images"]
        all_windows = []
        i = 0
        while i + hps.T <= len(images):
            all_windows.append(images[i : i + hps.T])
            i += hps.T - hps.overlap
        all_windows.append(images[i : len(images)])

        for window_idx, window_fnames in enumerate(all_windows):
            images = self.read_window(window_fnames)
            # s = self.synthesizer.synthesize_spectrograms(images)[0] ######
            model_output = infer_vid(images, self.synthesizer, mode="test")
            print("OUTPUT:", model_output.shape)


def to_sec(idx):
    frame_id = idx + 1
    sec = frame_id / float(hps.fps)
    return sec


def frames_generator(vidpath):
    frames = glob(os.path.join(vidpath, "*.jpg"))

    if len(frames) < hps.T:
        return

    yield frames


def load_model(ckpt_pth):
    device = torch.device("mps")

    checkpoint_dict = torch.load(ckpt_pth, map_location=device)["model"]
    model = MercuryNet()
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)

    model.load_state_dict(pretrained_dict)
    # for name, param in model.named_parameters():
    # 	print("Name", name)
    model = mode(model, True).eval()
    return model


def infer_vid(inputs, model, mode="train"):
    output = model.inference(inputs, mode)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_root", help="Speaker folder path", required=True)
    parser.add_argument(
        "-r", "--results_root", help="Speaker folder path", required=True
    )
    parser.add_argument(
        "--checkpoint", help="Path to trained checkpoint", required=True
    )
    parser.add_argument(
        "--preset", help="Speaker-specific hyper-params", type=str, required=False
    )
    args = parser.parse_args()

    # todo add speaker-specific parameters
    # with open(args.preset) as f:
    # 	sif.hparams.parse_json(f.read())

    # sif.hparams.set_hparam('eval_ckpt', args.checkpoint)
    videos = [
        os.path.join(args.data_root, f)
        for f in os.listdir(args.data_root)
        if os.path.isdir(os.path.join(args.data_root, f))
    ]

    if not os.path.isdir(args.results_root):
        os.mkdir(args.results_root)

    GTS_ROOT = os.path.join(args.results_root, "gts/")
    WAVS_ROOT = os.path.join(args.results_root, "wavs/")
    files_to_delete = []

    if not os.path.isdir(GTS_ROOT):
        os.mkdir(GTS_ROOT)
    else:
        files_to_delete = list(glob(GTS_ROOT + "*"))
    if not os.path.isdir(WAVS_ROOT):
        os.mkdir(WAVS_ROOT)
    else:
        files_to_delete.extend(list(glob(WAVS_ROOT + "*")))
    for f in files_to_delete:
        os.remove(f)

    tacotron = load_model(args.checkpoint)
    model = Generator(tacotron)

    template = "ffmpeg -y -loglevel panic -ss {} -i {} -to {} -strict -2 {}"
    for vid in videos:
        print("generating for VID:", vid)
        vidpath = vid + "/"
        for images in frames_generator(vidpath):
            sample = {}
            sample["images"] = images
            vidname = vid.split("/")[-1]
            outfile = "{}{}.wav".format(WAVS_ROOT, vidname)
            try:
                model.vc(sample, outfile)
            except KeyboardInterrupt:
                exit(0)
            except Exception as e:
                print(e)
                continue

            command = template.format(
                0,
                vidpath + "audio.wav",
                len(images),
                "{}{}.wav".format(GTS_ROOT, vidname),
            )

            subprocess.call(command, shell=True)
