# from pickletools import TAKEN_FROM_ARGUMENT1
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
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


def alif_otl():
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
    parser.add_argument('--pN', type=int,
        required=False, default=10, 
        help="The number of samples")
    parser.add_argument('--epoch', type=int,
        required=False, default=50, 
        help="The number of epoch")
    args = parser.parse_args()
    while len(sys.argv) > 1:
        sys.argv.pop()

    tts_noise=TTS()
    pN=args.pN
    text=args.text
    root=args.rootdir
    savedir=args.savedir
    gamma=args.gamma
    beta=args.beta
    alpha=args.alpha
    sample_rate=16000
    epoch=args.epoch
    lr=0.008
    threshold=0.4
    early_p=1
    total_count=0
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
        
    n=0
    while n < pN:
        all_loss=[]
        all_cer=[]
        # all_Linf=[]
        bestloss=99999
        noise=torch.autograd.Variable(torch.zeros(feature.shape)).to(device)
        noise.requires_grad = True
        opt=torch.optim.Adam([noise], lr=lr, betas=(0.5, 0.999))
        # opt = StepLR(optimizer, step_size=10, gamma=1)
        mel_noise=torch.tensor(np.random.uniform(-gamma, gamma, (1, 80, ori_mel.shape[-1]*10))).to(device)
        criterion = Tacotron2Loss()
        online_audio="online_attack_audio_"+str(n)+".wav"
        offline_audio="offline_attack_audio_"+str(n)+".wav"
        loss_name="loss_"+str(n)+".jpg"
        wav.write(os.path.join(root, savedir, online_audio), sample_rate, ori_audio)
        count=0
        offiter=0
        oniter=0
        right=True  # the decode is right or not
        for i in range(epoch):
            try: 
                print(f"epoch {i}")
                f=feature+noise
                _, adv_mel, adv_gate, _=tts_noise.model.inference2(f)
                # print(torch.mean(torch.abs(adv_mel)))

                adv_mel[:, 0: 30,:]*=alpha
                # add mel noise and set the low frequency to zero
                adv_mel+=mel_noise[:, :, :adv_mel.size(dim=-1)]
                adv_mel[:,: beta,:]=0

                opt.zero_grad()
                loss = criterion(ori_mel, ori_gate, adv_mel, adv_gate)
           
                all_loss.append(loss)
                loss.backward(retain_graph=True)
                opt.step()
                noise.requires_grad = False
                noise[: , -1, :]=0
                noise.requires_grad = True

                # This is for online attack
                if loss<bestloss:
                    # bestloss=loss
                    with torch.no_grad():
                        audio = tts_noise.waveglow.infer(adv_mel, sigma=0.666)[0].data.cpu().numpy()
                        # audio=np.array(audio/np.mean(np.abs(audio))*2000, dtype=np.int16)
                        # audio=np.array(audio/np.max(np.abs(audio)), dtype=np.float)
                        audio=np.array(audio/np.max(np.abs(audio))*10000, dtype=np.int16)
                        wav.write(os.path.join(root, savedir, "audio_cand.wav"), sample_rate, audio)
                    trans=get_trans(os.path.join(root, savedir), "audio_cand.wav")
                    print(trans)
                    cer=Levenshtein.distance(ori_trans, trans)/len(ori_trans)
                    print(cer)
                    count+=1
                    all_cer.append(cer)
                    if(cer==0):
                        os.system("cp "+os.path.join(root, savedir, "audio_cand.wav")+" "+os.path.join(root, savedir, online_audio))
                        # wav.write(os.path.join(root, savedir, online_audio), sample_rate, audio)
                        # wav.write(os.path.join(root, savedir, noisy_audio_name), sample_rate, noisy_audio)
                        bestloss=loss
                        oniter=i

                # This is for offline attack
                if early_stop(noise, threshold, p=early_p) or i==epoch-1:
                    with torch.no_grad():
                        audio = tts_noise.waveglow.infer(adv_mel, sigma=0.666)[0].data.cpu().numpy()
                        # audio=np.array(audio/np.mean(np.abs(audio))*2000, dtype=np.int16)
                        # audio=np.array(audio/np.max(np.abs(audio)), dtype=np.float)
                        audio=np.array(audio/np.max(np.abs(audio))*10000, dtype=np.int16)
                        wav.write(os.path.join(root, savedir, offline_audio), sample_rate, audio)
                    offiter=i
                    break
            except Exception as e:
                print(e)
                right=False
                break
        if not right:
            continue
        total_count+=count        
        print("*"*100)
        online_trans=get_trans(os.path.join(root, savedir), online_audio)
        print(online_trans)
        offline_trans=get_trans(os.path.join(root, savedir), offline_audio)
        print(offline_trans)
        with open(os.path.join(root, savedir, report_name), "a") as report:
            report.write("*"*100+"\n")
            report.write(f"online_audio{n}: {online_trans}\n")
            report.write(f"offline_audio{n}: {offline_trans}\n")
            report.write(f"#query{n}: {count}\n")
            report.write(f"#oniter{n}: {oniter}\n")
            report.write(f"#offiter{n}: {offiter}\n")
            report.write("*"*100+"\n")
            if n==pN-1:
                report.write(f"#total_query: {total_count}\n")
        x=range(len(all_loss))
        # plt.figure()
        plt.title('loss tendency')
        plt.xlabel('iter')
        plt.ylabel('loss')
        plt.plot(x,np.array(all_loss))
        plt.savefig(os.path.join(root, savedir, loss_name))
        n+=1

def early_stop(noise, threshold=0.6, p=1):
    if torch.sum((abs(noise)>threshold).float()) >p: # * (noise.shape[1]*noise.shape[2]):
        return True
    return False

if __name__=="__main__":
    alif_otl()