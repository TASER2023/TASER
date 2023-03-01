import sys
sys.path.append('tacotron2/')
sys.path.append('tacotron2/waveglow/')
import numpy as np
import torch

from hparams import create_hparams
from train import load_model
from text import text_to_sequence
from denoiser import Denoiser

from scipy.io import wavfile as wav
import os
from torch import nn
import Levenshtein
import argparse

from ASRs import get_trans_amazon, get_trans_azure, get_trans_google, get_trans_iflytek, get_trans_tencent
from optimizer import POS2

device=torch.device('cuda')

class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    def forward(self, ori_mel, ori_gate, adv_mel, adv_gate):
        ori_gate = ori_gate.view(-1, 1)
        adv_gate = adv_gate.view(-1, 1)
        len = ori_gate.shape[0] if ori_gate.shape[0] < adv_gate.shape [0] else adv_gate.shape [0]
        mel_loss = (nn.MSELoss()(adv_mel[:, :, 0:len], ori_mel[:, :, 0:len]))
        gate_loss = nn.BCEWithLogitsLoss()(adv_gate[0:len, :], ori_gate[0:len, :])
        print(-mel_loss.item(), gate_loss.item())
        return (-mel_loss + gate_loss)

class TTS():
    def __init__(self):
        self.hparams = create_hparams()
        self.hparams.sampling_rate =16000#   22050

        checkpoint_path = "tacotron2/tacotron2_statedict.pt"
        self.model = load_model(self.hparams)
        self.model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
        _ = self.model.to(device).train() # .cuda().eval()# .half()
        for module in self.model.modules():
            if isinstance(module, torch.nn.modules.BatchNorm1d):
                module.eval()
        print("device count:",torch.cuda.device_count())
        print("current_select_device:", torch.cuda.current_device())
        waveglow_path = 'tacotron2/waveglow_256channels_universal_v5.pt'
        self.waveglow = torch.load(waveglow_path)['model']
        self.waveglow.to(device).eval() # .cuda().eval()# .half()
        for k in self.waveglow.convinv:
            k.float()
        self.denoiser = Denoiser(self.waveglow)

    def get_feature(self, text):
        sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
        sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
        return self.model.get_feature2(sequence)

    def get_mels_gates(self, features):
        # _, adv_mel, adv_gate, _=tts_noise.model.inference2(f)
        adv_mels=[]
        adv_gates=[]
        for feature in features:
            f = torch.autograd.Variable(torch.from_numpy(np.array([feature]))).to(device).float()
            _, adv_mel, adv_gate, _=self.model.inference2(f)
            adv_mels.append(adv_mel)
            adv_gates.append(adv_gate)
        return adv_mels, adv_gates

def taser_ota():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--text', type=str,
        required=True, default="", 
        help="The target text")
    parser.add_argument('--rootdir', type=str,
        required=True, default="", 
        help="The name of root dir")
    parser.add_argument('--savedir', type=str,
        required=True, default="", 
        help="The name of output dir")
    parser.add_argument('--ASR', type=str,
        required=True, default="tencent", 
        help="The ASR to use")
    # parser.add_argument('--threshold', type=float,
    #     required=True, default=1, 
    #     help="The threshold of the noise, if you choose l1/l8 mode, then it means the average/max abs amplitude of every point in the noise")
    parser.add_argument('--gamma', type=float,
        required=False, default=2, 
        help="The max amplitude of the noise add to mel spctrogram")
    parser.add_argument('--beta', type=int,
        required=False, default=2, 
        help="How many rows of the mel sepctrogram should be replaced by others (row of 0:beta)")
    parser.add_argument('--alpha', type=float,
        required=False, default=1, 
        help="The mel will * alpha")
    parser.add_argument('--eta', type=float,
        required=False, default=0, 
        help="The white noise will * eta")
    parser.add_argument('--pN', type=int,
        required=False, default=20, 
        help="The number of particles")
    parser.add_argument('--epoch', type=int,
        required=False, default=10, 
        help="The number of epoch")
    args = parser.parse_args()
    while len(sys.argv) > 1:
        sys.argv.pop()

    tts_noise=TTS()
    pN=args.pN
    text=args.text  # "Read my new messages."
    root=args.rootdir
    savedir=args.savedir  # "bigtest9"
    gamma=args.gamma
    beta=args.beta
    alpha=args.alpha
    eta=args.eta
    epoch=args.epoch
    sample_rate=16000
    lr=0.008
    threshold=0.4
    early_p=1
    report_name="report.txt"
    get_trans=None
    if args.ASR=="tencent":
        get_trans=get_trans_tencent
    if args.ASR=="iflytek":
        get_trans=get_trans_iflytek    
    elif args.ASR=="google": 
        get_trans=get_trans_google
    elif args.ASR=="amazon": 
        get_trans=get_trans_amazon
    elif args.ASR=="azure": 
        get_trans=get_trans_azure
    if not os.path.exists(os.path.join(root, savedir)):
        os.makedirs(os.path.join(root, savedir))
    with open(os.path.join(root, savedir, report_name), "w") as report:
        report.write(f"original text: {text}\n")
        report.write(f"learning rate: {lr}\n")
        report.write(f"threshold: {threshold}\n")
        report.write(f"early p: {early_p}\n")
        report.write(f"gamma: {gamma}\n")
        report.write(f"beta: {beta}\n")
        report.write(f"ASR: {args.ASR}\n")
    feature=tts_noise.get_feature(text)
    _, ori_mel, ori_gate, _=tts_noise.model.inference2(feature)
    ori_gate=(ori_gate>0.5).float()
    
    with torch.no_grad():
        ori_audio = tts_noise.waveglow.infer(ori_mel, sigma=0.666)[0].data.cpu().numpy()
        # ori_audio=np.array(ori_audio/np.mean(np.abs(ori_audio))*2000, dtype=np.int16)
        # ori_audio=np.array(ori_audio/np.max(np.abs(ori_audio)), dtype=np.float)
        ori_audio=np.array(ori_audio/np.max(np.abs(ori_audio))*10000, dtype=np.int16)
        wav.write(os.path.join(root, savedir, "original_audio.wav"), sample_rate, ori_audio)
    ori_trans=get_trans(os.path.join(root, savedir), "original_audio.wav")
    print(ori_trans)
    
    noise=np.zeros((pN, feature.shape[1], feature.shape[2]))
    noise[: , -1, :]=0
    mel_noise=torch.tensor(np.random.uniform(-gamma, gamma, (1, 80, ori_mel.shape[-1]*10))).to(device)
    pos=POS2(pN, feature.shape[1], feature.shape[2], noise)
    criterion = Tacotron2Loss()
    for e in range(epoch):
        print(f"**********epoch {e}**********")
        with open(os.path.join(root, savedir, "report.txt"), "a") as report:
            report.write(f"**********epoch {e}**********\n")
        f=np.array([feature[0].cpu().detach().numpy()]*pN)
        # print(feature[0].shape, f.shape, noise.shape)
        f+=noise
        # _, adv_mel, adv_gate, _=tts_noise.model.inference2(f)
        adv_mels, adv_gates=tts_noise.get_mels_gates(f)
        audios=[]
        for adv_mel in adv_mels:
            adv_mel[:, 0: 30,:]*=alpha
            adv_mel+=mel_noise[:, :, :adv_mel.size(dim=-1)]
            adv_mel[:,: beta,:]=0
            audio=tts_noise.waveglow.infer(adv_mel, sigma=0.666)[0].data.cpu().numpy()
            # audio=np.array(audio/np.max(np.abs(audio)), dtype=np.float)
            audio=np.array(audio/np.max(np.abs(audio))*10000, dtype=np.int16)
            audios.append(audio)
        noisy_audios=strengthen(audios, eta)
        inf=100000
        loss=[]
        cers=[]
        for i in range(len(audios)):
            audio_name="audio_cand.wav"
            noisy_name="noisy_audio_cand.wav"
            wav.write(os.path.join(root, savedir, "audio_cand.wav"), sample_rate, np.array(audios[i]))
            wav.write(os.path.join(root, savedir, "noisy_audio_cand.wav"), sample_rate, np.array(noisy_audios[i]))
            trans=get_trans(os.path.join(root, savedir), audio_name)
            print("trans:", trans)
            cer1=Levenshtein.distance(ori_trans, trans)/len(ori_trans)
            trans=get_trans(os.path.join(root, savedir), noisy_name)
            print("noisy:", trans)
            cer2=Levenshtein.distance(ori_trans, trans)/len(ori_trans)
            l = criterion(ori_mel, ori_gate, adv_mels[i], adv_gates[i])
            if cer1==0 and cer2 ==0:
                loss.append(l.data.cpu().numpy())
                wav.write(os.path.join(root, savedir,"audio"+str(e)+"_"+str(i)+".wav"), sample_rate, np.array(audios[i]))
                wav.write(os.path.join(root, savedir,"noisy_audio"+str(e)+"_"+str(i)+".wav"), sample_rate, np.array(noisy_audios[i]))
            else:
                loss.append(inf)
            cers.append((cer1, cer2))
        print("cer:",cers)            
        print("loss:",loss)
        noise, bestparam, bestidx, bestflag=pos.update(loss)
        noise[:, -1, :]=0
        with open(os.path.join(root, savedir, "report.txt"), "a") as report:
            report.write(f"cer:{cers}\n")
            report.write(f"loss:{loss}\n")
        if bestflag:
            wav.write(os.path.join(root, savedir,"final_audio.wav"), sample_rate, np.array(audios[bestidx]))
            wav.write(os.path.join(root, savedir,"final_noisy_audio.wav"), sample_rate, np.array(noisy_audios[bestidx]))
            with open(os.path.join(root, savedir, "report.txt"), "a") as report:
                report.write(f"*****update at epoch {e}, the {bestidx}th audio*****\n")
            bestloss=loss[bestidx]
            

def add_noise(audio, eta):
    noise=np.random.standard_normal(size=len(audio))
    noise=noise/np.max(np.abs(noise))*eta*10000
    noisy_audio=np.array((audio+noise)/np.max(np.abs(audio+noise))*10000, dtype=np.int16)
    return noisy_audio

def strengthen(audios, eta):
    strengthen_audios=[]
    for audio in audios:
        # strengthen_audios.append(add_noise(add_rir(audio)))
        strengthen_audios.append(add_noise(audio, eta))
    return strengthen_audios

def early_stop(noise, threshold=0.6, p=1):
    if torch.sum((abs(noise)>threshold).float()) >p: # * (noise.shape[1]*noise.shape[2]):
        return True
    return False

if __name__=="__main__":
    taser_ota()