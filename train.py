import os
import torch
from pyrealsense2.pyrealsense2 import device

from evaluate import eval_model, create_lr_scheduler
from utils.data_loading import BasicDataset
import utils.transforms as T
from model import ThreeDJAUNet3Plus
from loss.dice_coefficient_loss import dice_loss, build_target
from tqdm import tqdm


def _iou(pred, target, size_average=True):
	b = pred.shape[0]
	IoU = 0.0
	for i in range(0, b):
		# compute the IoU of the foreground
		Iand1 = torch.sum(target[i, :, :, :] * pred[i, :, :, :])
		Ior1 = torch.sum(target[i, :, :, :]) + torch.sum(pred[i, :, :, :]) - Iand1
		IoU1 = Iand1 / Ior1
		
		# IoU loss is (1-IoU1)
		IoU = IoU + (1 - IoU1)
	
	return IoU / b


class IOU(torch.nn.Module):
	def __init__(self, size_average=True):
		super(IOU, self).__init__()
		self.size_average = size_average
	
	def forward(self, pred, target):
		return _iou(pred, target, self.size_average)


def iou_loss(pred, label):
	iou_loss = IOU(size_average=True)
	iou_out = iou_loss(pred, label)
	# print("iou_loss:", iou_out.data.cpu().numpy())
	return iou_out


def _iou(pred, target, size_average=True):
	b = pred.shape[0]
	IoU = 0.0
	for i in range(0, b):
		#compute the IoU of the foreground
		Iand1 = torch.sum(target[i, :, :, :] * pred[i, :, :, :])
		Ior1 = torch.sum(target[i, :, :, :]) + torch.sum(pred[i, :, :, :]) - Iand1
		IoU1 = Iand1 / Ior1
		
		#IoU loss is (1-IoU1)
		IoU = IoU + (1 - IoU1)
	
	return IoU / b


class IOU(torch.nn.Module):
	def __init__(self, size_average=True):
		super(IOU, self).__init__()
		self.size_average = size_average
	
	def forward(self, pred, target):
		return _iou(pred, target, self.size_average)


def iou_loss(pred, label):
	iou_loss = IOU(size_average=True)
	iou_out = iou_loss(pred, label)
	return iou_out


def criterion(inputs, target, loss_weight=None, num_classes: int = 2, dice: bool = True, ignore_index: int = -100,
			  loss_weights=[]):
	losses = {}
	if isinstance(inputs, dict):
		for name, x in inputs.items():
			loss = torch.nn.functional.cross_entropy(x, target, ignore_index=ignore_index, weight=loss_weight)
			if dice is True:
				dice_target = build_target(target, num_classes, ignore_index)
				loss += dice_loss(x, dice_target, multiclass=True, ignore_index=ignore_index)
			losses[name] = loss
	elif isinstance(inputs, tuple):
		for i, x in enumerate(inputs):
			loss = torch.nn.functional.cross_entropy(x, target, ignore_index=ignore_index, weight=loss_weight)
			if dice is True:
				dice_target = build_target(target, num_classes, ignore_index)
				loss += dice_loss(x, dice_target, multiclass=True, ignore_index=ignore_index)
			losses[f'out{i}'] = loss
		# d1,d2,d3,d4,d5 weighted average
		weights = loss_weights
		losses['out'] = 0
		for i in range(len(weights)):
			losses['out'] += weights[i] * losses[f'out{i}']
		losses['out'] = losses['out'] / sum(weights)
	else:
		losses['out'] = torch.nn.functional.cross_entropy(inputs, target, ignore_index=ignore_index, weight=loss_weight)
	
	return losses['out']


class SegmentationPresetTrain:
	"""
	Preprocessing for training
	"""
	
	def __init__(self, base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5,
				 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
		min_size = int(0.5 * base_size)
		max_size = int(1.2 * base_size)
		
		trans = [T.RandomResize(min_size, max_size)]
		if hflip_prob > 0:
			trans.append(T.RandomHorizontalFlip(hflip_prob))
		if vflip_prob > 0:
			trans.append(T.RandomVerticalFlip(vflip_prob))
		trans.extend([
			T.RandomCrop(crop_size),
			T.ToTensor(),
			T.Normalize(mean=mean, std=std),
		])
		self.transforms = T.Compose(trans)
	
	def __call__(self, img, target):
		return self.transforms(img, target)


class SegmentationPresetEval:
	def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
		self.transforms = T.Compose([
			T.ToTensor(),
			T.Normalize(mean=mean, std=std),
		])
	
	def __call__(self, img, target):
		return self.transforms(img, target)


def get_transform(train, image_size=512, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
	base_size = 512
	crop_size = image_size
	
	if train:
		return SegmentationPresetTrain(base_size, crop_size, mean=mean, std=std)
	else:
		return SegmentationPresetEval(mean=mean, std=std)


def main(args):
	device = torch.device(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
	# device = torch.device("cpu")
	batch_size = args.batch_size
	# segmentation nun_classes + background
	num_classes = args.num_classes + 1
	train_dataset = BasicDataset(images_dir=args.image_path, mask_dir=args.mask_path)
	num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
	train_loader = torch.utils.data.DataLoader(train_dataset,
											   batch_size=batch_size,
											   num_workers=num_workers,
											   shuffle=False,
											   pin_memory=False,
											   )
	
	model = ThreeDJAUNet3Plus(in_channels=3, n_classes=num_classes, PCM=True)
	total = sum([param.nelement() for param in model.parameters()])
	print("====================Number of parameter:%.2fM====================" % (total / 1e6))
	
	model.to(device)
	params_to_optimize = [p for p in model.parameters() if p.requires_grad]
	
	if args.optimizer == 'SGD-M':
		optimizer = torch.optim.SGD(
			params_to_optimize,
			lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
		)
	else:
		optimizer = torch.optim.Adam(params_to_optimize, lr=args.lr, weight_decay=args.weight_decay)
	if args.checkpoint is not None:
		model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
		model.to(device)
		print("load checkpoint from %s" % args.checkpoint)
	epochs = args.epochs
	model.train()
	for epoch in range(epochs):
		print(f"Epoch {epoch + 1}/{epochs}")
		bar = tqdm(train_loader)
		for batch in bar:
			imgs = batch["image"]
			true_masks = batch["mask"]
			imgs = imgs.to(device=device, dtype=torch.float32)
			mask_type = torch.float32
			true_masks = true_masks.to(device=device, dtype=mask_type)
			true_masks = torch.nn.functional.one_hot(true_masks.long(), num_classes=num_classes).permute(0, 3, 2, 1)
			masks_pred = model(imgs)
			losses = []
			d0_loss = 0
			for i in range(6):
				loss = dice_loss(masks_pred[i], true_masks.float()) \
							+ torch.nn.CrossEntropyLoss()(masks_pred[i], true_masks.float())
				loss.to(device=device).to(torch.float32)
				losses.append(loss)
				if not i:
					d0_loss = loss
			loss = sum(losses)
			print("d0_loss", d0_loss)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			bar.set_description(f"loss: {loss.item():.5f}")
			bar.set_description(f"Epoch {epoch + 1}/{epochs}")
			del loss
		torch.save(model.state_dict(), args.save_path + f"unet3_epoch{epoch + 1}.pt")
		print(f"Checkpoint {epoch + 1} saved !")
	print("Training finished")


def parse_args():
	import argparse
	parser = argparse.ArgumentParser(description="pytorch unet training")
	parser.add_argument("--image_path", default="/home/ibra/Documents/DATA/segmentation/datasets/branch/test1/imgs/",
						help="dataset root")
	parser.add_argument("--mask_path", default="/home/ibra/Documents/DATA/segmentation/datasets/branch/test1/masks/",
						help="dataset root")
	parser.add_argument("--num_classes", default=1, type=int)
	parser.add_argument("--device", default="cuda:0", help="training device")
	parser.add_argument("-b", "--batch_size", default=1, type=int)
	parser.add_argument("--epochs", default=100, type=int, metavar="N",
						help="number of total epochs to train")
	parser.add_argument('--lr', default=0.00001, type=float, help='initial learning rate')
	parser.add_argument('--optimizer', default='SGD-M', choices=['SGD-M', 'Adam'], help='optimizer')
	parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
	parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
						metavar='W', help='weight decay (default: 1e-4)',
						dest='weight_decay')
	# Mixed precision training parameters
	parser.add_argument("--amp", default=False, type=bool, help="Use torch.cuda.amp for mixed precision training")
	parser.add_argument("--image-csize", default=320, type=int, help="size fo input image ")
	parser.add_argument("--checkpoint", default=None, help="resume from checkpoint, None means training from scratch")
	parser.add_argument('--save_path', default="./save_weights/", help="save path")
	args = parser.parse_args()
	
	return args


if __name__ == '__main__':
	args = parse_args()
	
	if not os.path.exists("./save_weights"):
		os.mkdir("./save_weights")
	main(args)
