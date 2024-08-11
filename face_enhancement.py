from swinIR.swinir import *
import matplotlib.pyplot as plt
import os

#upscale=2, in_chans=3, img_size=64, window_size=8, img_range=1., 
#depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv'
# Ok, let's load the model now.
model = SwinIR(upscale=2, in_chans=3, img_size=64, window_size=8, img_range=1.,
               depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
               mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')

param_key_g = "params"
pretrained_dict = torch.load("swinIR/swin-ir.pth")

model.load_state_dict(pretrained_dict[param_key_g] if param_key_g in pretrained_dict.keys() else pretrained_dict, strict=True)

model = model.eval()
# model = model.to("cuda") # If you have cuda enabled.

def inference_swin_ir(input_image_path):
    img_lq = cv2.imread(input_image_path, cv2.IMREAD_COLOR).astype(np.float32) / 255
    img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
    img_lq = torch.from_numpy(img_lq).float().unsqueeze(0) #.to("cuda") # add .to() if u have cuda 
    window_size = 8
    SCALE = 2
    with torch.no_grad():
        # pad input image to be a multiple of window_size
        _, _, h_old, w_old = img_lq.size()
        h_pad = (h_old // window_size + 1) * window_size - h_old
        w_pad = (w_old // window_size + 1) * window_size - w_old
        img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
        img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
        output = model(img_lq)
        output = output[..., :h_old * SCALE, :w_old * SCALE]
    output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = (output * 255.0).round().astype(np.uint8)
    output = output.transpose(1, 2, 0)
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    return output
    
# directory : objects_and_faces_detected/frame_{i}/*.jpg ------

inp_dir = 'objects_and_faces_detected'
out_dir = 'enhanced_faces_swinir'
os.makedirs(out_dir, exist_ok=True)

# traverse input directory, but first find no. of folders in the directory
n = 0
for folder in os.listdir(inp_dir):
    n+=1

for i in range(n):
    inp_path = os.path.join(inp_dir, f'frame_{i}')
    out_path = os.path.join(out_dir, f'frame_{i}')
    os.makedirs(out_path, exist_ok=True)
    for file in os.listdir(inp_path):
        # if not jpg, continue
        if not file.endswith('.jpg'):
            continue
        inp_file_path = os.path.join(inp_path, file)
        out_file_path = os.path.join(out_path, file)
        op = inference_swin_ir(inp_file_path)
        cv2.imwrite(out_file_path, op)
