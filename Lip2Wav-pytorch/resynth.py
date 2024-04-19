import numpy as np

# constants
AUDIO_SR = 22050
VID_FRAME_RATE = 30
HOP_SIZE = AUDIO_SR // VID_FRAME_RATE
assert((22050/VID_FRAME_RATE) % 1 == 0)

def resynthesize(features, max_length=None):
    '''
    resynthesize audio from features
    expects features = (features, time) array, with features being stacked (f0, voiced, rms)
    '''
    TOTAL_LEN = features.shape[-1]*HOP_SIZE if max_length==None else min(features.shape[-1]*HOP_SIZE, max_length)
    omega = np.zeros_like(TOTAL_LEN)
    for i in range(features.shape[-1] - 1):
        last_omega = omega[i*HOP_SIZE - 1]
        omega[i*HOP_SIZE : (i+1)*HOP_SIZE] = np.arange(HOP_SIZE)/AUDIO_SR*2*np.pi * features[0,i] + last_omega
    
    amplitudes = np.interp(x = np.linspace(0, 1, num=features.shape[-1]*HOP_SIZE),
                           xp = np.linspace(0, 1, num=features.shape[-1]),
                           fp = features[2,:])
    
    return np.sin(omega)[:TOTAL_LEN]**5 * amplitudes[:TOTAL_LEN]