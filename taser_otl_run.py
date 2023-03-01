import os

# ASRs=["tencent", "azure", "iflytek", "amazon", "google"]
# commands=["\"Airplane mode on.\"", "\"Call one two three.\"", "\"Cancel my alarm clock.\"", "\"Darn it!\"", "\"I can't take it anymore!\"", "\"I need help.\"", "\"Navigate to my office.\"", "\"Send a message to my mom.\"", "\"Transfer the payment.\"", "\"Turn on the light.\"",  "\"Unlock the door.\"", "\"What's the time?\""]
# mels=[("0", "0"), ("1", "1"), ("1", "2"), ("2", "4")]
# alphas= ["0.25", "0.3", "1"]

'''
set the parameters here
Note that "beta" here is different from that in the paper. 
"beta" here denotes the index of Mel filterbank.
'''
ASR="tencent"
command="\"What's the time?\""  
mel=("1", "1") # (gamma, beta)
alpha="0.3"

rootdir="taser_otl_example/"+ASR+"_alpha"+alpha+"_beta"+mel[1]+"_gamma"+mel[0]
savedir=command.replace(" ", "_").replace(".", "").replace("!", "").replace("?", "").replace("'", "").lower()
cline="python taser_otl.py --text "+command+" --rootdir " +rootdir+ " --savedir "+savedir+" --ASR "+ASR+" --gamma "+mel[0]+" --beta "+mel[1]+" --alpha "+alpha
print(cline)
os.system(cline)