import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from utils.data_loading import BasicDataset
from inference_model import ThreeDJAUNet3Plus
import cv2


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.0):

	img = torch.from_numpy(BasicDataset.preprocess(None,
	                                               full_img,
	                                               scale_factor,
	                                               is_mask=False))
	img = img.unsqueeze(0).to(device = device, dtype=torch.float32)
	with torch.no_grad():
		output = net(img)
		probs = F.softmax(output, dim=1).argmax(dim=1).permute(2, 1, 0).squeeze(2)
		print("probs.shape:", probs.shape)
		print("probs:", probs)
	return probs


def get_args():
	parser = argparse.ArgumentParser(description='Predict masks from input images')
	parser.add_argument('--model', '-m', default='/home/ibra/Documents/3DJA-UNet3Plus/save_weights/unet3_epoch17.pt', metavar='FILE',
	                    help='Specify the file in which the model is stored')
	parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
	                    help='Minimum probability value to consider a mask pixel white')
	parser.add_argument('--scale', '-s', type=float, default=1,
	                    help='Scale factor for the input images')
	parser.add_argument('--img', '-i', default='/home/ibra/Documents/DATA/segmentation/datasets/branch/test/image/img_985239177n.png', help='Filenames of input images')
	return parser.parse_args()


if __name__ == '__main__':
	args = get_args()
	net = ThreeDJAUNet3Plus(in_channels=3, n_classes=2)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	net.load_state_dict(torch.load(args.model, map_location=device), strict=False)
	net.eval()
	net.to(device=device)
	name = args.img
	img = Image.open(name)
	img = img.convert('RGB')
	mask = predict_img(net=net,
	                   full_img=img,
	                   scale_factor=args.scale,
	                   out_threshold=args.mask_threshold,
	                   device=device)
	mask = mask.cpu().numpy()
	if mask.sum() == 0:
		print('no mask')
		exit(0)
		
	mask = mask * 255
	# mask = mask.astype(np.uint8)
	print("mask", mask)
	print("mask shape", mask.shape)
	mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
	# result = mask_to_image(mask)
	img2 = cv2.imread(name)
	img2 = cv2.resize(img2, (720, 720))
	img_mask = cv2.resize(mask, (720, 720))
	merge_alpha_compose = cv2.addWeighted(img2, 0.5, img_mask, 0.5, 0)
	merge_alpha_compose = np.vstack((cv2.resize(img_mask, (720, 720)),
	                                 cv2.resize(img2, (720, 720)),
	                                 merge_alpha_compose))
	merge_alpha_compose = cv2.resize(merge_alpha_compose, (720, 1500))
	while True:
		cv2.imshow('result', merge_alpha_compose)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break


