from __future__ import division
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
import random
import cv2

in1k_mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
in1k_std =  torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)  

def color_offset(value,min,max,ratio=0.08):
    offset = np.random.uniform(1-ratio, 1+ratio)
    value = value*offset
    value = np.clip(value,min,max)
    return value

class RandomContrast(object):
    """Random contrast"""
    def __init__(self,color_diff):
        self.color_diff = color_diff

    def __call__(self, sample):
        if np.random.random() < 0.5:
            contrast_factor = np.random.uniform(0.8, 1.2)
            sample['left'] = F.adjust_contrast(sample['left'], contrast_factor)
            if not self.color_diff:
                sample['right'] = F.adjust_contrast(sample['right'], contrast_factor)
            else:
                if np.random.random() < 0.5:
                    contrast_factor2 = color_offset(contrast_factor,0.8,1.2)
                else:
                    contrast_factor2 = contrast_factor
                sample['right'] = F.adjust_contrast(sample['right'], contrast_factor2)
        return sample

class RandomGamma(object):
    def __init__(self,color_diff):
        self.color_diff = color_diff
        
    def __call__(self, sample):
        if np.random.random() < 0.5:
            gamma = np.random.uniform(0.7, 1.5)  # adopted from FlowNet

            sample['left'] = F.adjust_gamma(sample['left'], gamma)
            if not self.color_diff:
                sample['right'] = F.adjust_gamma(sample['right'], gamma)
            else:
                # gamma2 = color_offset(gamma,0.7,1.5)
                if np.random.random() < 0.5:
                    gamma2 = color_offset(gamma,0.7,1.5)
                else:
                    gamma2 = gamma
                sample['right'] = F.adjust_gamma(sample['right'], gamma2)

        return sample


class RandomBrightness(object):
    def __init__(self,color_diff):
        self.color_diff = color_diff

    def __call__(self, sample):
        if np.random.random() < 0.5:
            brightness = np.random.uniform(0.5, 2.0) # 0.5 2.0

            sample['left'] = F.adjust_brightness(sample['left'], brightness)
            if not self.color_diff:
                # brightness2 = color_offset(brightness,0.5,2.0,0.1)
                # print('brightness',brightness)
                # print('brightness2',brightness2)
                # sample['right'] = F.adjust_brightness(sample['right'], brightness2)
                sample['right'] = F.adjust_brightness(sample['right'], brightness)
            else:
                if np.random.random() < 0.5:
                    brightness2 = color_offset(brightness,0.5,2.0)
                else:
                    brightness2 = brightness
                sample['right'] = F.adjust_brightness(sample['right'], brightness2)

        return sample


class RandomHue(object):
    def __init__(self,color_diff):
        self.color_diff = color_diff

    def __call__(self, sample):
        if np.random.random() < 0.5:
            hue = np.random.uniform(-0.1, 0.1)

            sample['left'] = F.adjust_hue(sample['left'], hue)
            if not self.color_diff:
                sample['right'] = F.adjust_hue(sample['right'], hue)
            else:
                if np.random.random() < 0.5:
                    hue2 = color_offset(hue,-0.1,0.1)
                else:
                    hue2 = hue
                sample['right'] = F.adjust_hue(sample['right'], hue2)

        return sample


class RandomSaturation(object):
    def __init__(self,color_diff):
        self.color_diff = color_diff

    def __call__(self, sample):
        if np.random.random() < 0.5:
            saturation = np.random.uniform(0.8, 1.2)
            sample['left'] = F.adjust_saturation(sample['left'], saturation)
            if not self.color_diff:
                sample['right'] = F.adjust_saturation(sample['right'], saturation)
            else:
                if np.random.random() < 0.5:
                    saturation2 = color_offset(saturation,0.8,1.2)
                else:
                    saturation2 = saturation
                sample['right'] = F.adjust_saturation(sample['right'], saturation2)
        return sample

class _RandomColor(object):
    def __init__(self,color_diff):
        self.transforms = [RandomContrast(color_diff),
                      RandomGamma(color_diff),
                      RandomBrightness(color_diff),
                      RandomHue(color_diff),
                      RandomSaturation(color_diff)]
        # self.transforms = [RandomBrightness(color_diff)] 
    
    def __call__(self, img1, img2,disp,append):
        sample = {'left':Image.fromarray(img1),'right':Image.fromarray(img2)}
        if np.random.random() < 0.5:
            # A single transform
            t = random.choice(self.transforms)
            sample = t(sample)
        else:
            # Combination of transforms in Random order
            random.shuffle(self.transforms)
            # print('-----')
            for t in self.transforms:
                sample = t(sample)
                # print('type transformer',t)
            # print('-----')
        return np.array(sample['left']).astype(np.float32),np.array(sample['right']).astype(np.float32),disp,append
        # return np.array(sample['left']).astype(np.float32),np.array(sample['right']).astype(np.float32),disp
    
class Augmentor:
    def __init__(self, resize = None,RandomColor = False, rotate = False, scale = False, crop = None, HFlip = False, VFlip = False, norm = True, seed=0, color_diff=False,erase=False):
        super().__init__()
        self.resize = resize
        self.crop_size = crop
        self.scale_prob = 0.5
        self.scale_min = 0.6 # for scale w/o crop
        self.scale_max = 1.0 # for scale w/o crop
        self.scale_xonly = True # for scale w/ crop
        self.lminscale = 0.0 # for scale w/ crop
        self.lmaxscale = 0.5 # for scale w/ crop
        self.hminscale = -0.2 # for scale w/ crop
        self.hmaxscale = 0.4 # for scale w/ crop
        self.erase_prob = 0.5 # for scale w/ crop
        self.lhth = 800 # for scale w/ crop
        self.rng = np.random.RandomState(seed)
        self.transforms = []
                
        print('transforms: RandomColor:',RandomColor, ', norm:', norm, ', rotate:', rotate, ', scale:', scale, ', crop:', crop,', VFlip:', VFlip, ', resize:',resize, ', HFlip:', HFlip, ', color_diff:',color_diff,', erase:',erase)
               
        if not resize is None:
            if isinstance(resize, list):
                assert len(resize.shape) == 2
                assert resize[1] > resize[0], 'width of resize shape should large that the height?'
                self.transforms.append(self._resize_SizeBase) 
            else:
                assert 0<resize and resize<2, 'resize should be reasonable'
                self.transforms.append(self._resize__ScaleBase) 
        if RandomColor:
            self.transforms.append(_RandomColor(color_diff)) 
        if scale:
            self.transforms.append(self._random_scale_crop) 
            # if self.crop_size is None:
            #     self.transforms.append(self._random_scale) 
            # else:
            #     self.transforms.append(self._random_scale_crop) 
        if not self.crop_size is None:
            self.transforms.append(self._random_crop) 
        if HFlip:
            self.transforms.append(self._HorizontalFlip) 
        if VFlip:
            self.transforms.append(self._VerticalFlip) 
        if rotate:
            self.transforms.append(self._random_rotate) 
        if erase:
            self.transforms.append(self._random_erase) 
        self.transforms.append(self._img_to_tensor) 
        if norm:
            self.transforms.append(self._norm) 

    def _random_erase(self, img1, img2,disp,append):
        ht, wd = img1.shape[:2]
        if np.random.rand() < self.erase_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(50, 100)
                dy = np.random.randint(50, 100)
                img2[y0:y0+dy, x0:x0+dx, :] = mean_color
        return img1, img2, disp, append

    def _resize_SizeBase(self,img1,img2,disp,append=None):
        img1 = cv2.resize(img1,(self.resize[1],self.resize[0]),interpolation=cv2.INTER_LINEAR)
        img2 = cv2.resize(img2,(self.resize[1],self.resize[0]),interpolation=cv2.INTER_LINEAR)
        disp_scale = (float(self.resize[1]) / float(disp.shape[-1]))
        disp = cv2.resize(disp*disp_scale,(self.resize[1],self.resize[0]),interpolation=cv2.INTER_NEAREST)
        if not append is None:
            new = {}
            for i in append:
                new[i] = cv2.resize(append[i],(self.resize[1],self.resize[0]),interpolation=cv2.INTER_LINEAR)
            append = new
        return img1,img2,disp, append
    
    def _resize__ScaleBase(self,img1,img2,disp,append=None):
        img1 = cv2.resize(img1,None,fx=self.resize,fy=self.resize,interpolation=cv2.INTER_LINEAR)
        img2 = cv2.resize(img2,None,fx=self.resize,fy=self.resize,interpolation=cv2.INTER_LINEAR)
        disp = cv2.resize(disp*self.resize,None,fx=self.resize,fy=self.resize,interpolation=cv2.INTER_LINEAR)
        if not append is None:
            new = {}
            for i in append:
                new[i] = cv2.resize(append[i]*self.resize,None,fx=self.resize,fy=self.resize,interpolation=cv2.INTER_LINEAR)
            append = new
        return img1,img2,disp, append

    def _random_rotate(self,img1,img2,disp,append=None):
        if self.rng.binomial(1, 0.5):
            angle, pixel = 0.1, 2
            px = self.rng.uniform(-pixel, pixel)
            ag = self.rng.uniform(-angle, angle)
            image_center = (
                self.rng.uniform(0, img2.shape[0]),
                self.rng.uniform(0, img2.shape[1]),
            )
            rot_mat = cv2.getRotationMatrix2D(image_center, ag, 1.0)
            img2 = cv2.warpAffine(
                img2, rot_mat, img2.shape[1::-1], flags=cv2.INTER_LINEAR
            )
            trans_mat = np.float32([[1, 0, 0], [0, 1, px]])
            img2 = cv2.warpAffine(
                img2, trans_mat, img2.shape[1::-1], flags=cv2.INTER_LINEAR
            )
        return img1,img2,disp, append
     
    # scale w/ crop
    def _random_scale_crop(self, img1, img2, disp,append=None):
        ch,cw = self.crop_size
        h,w = img1.shape[:2]        
        if self.scale_prob>0. and np.random.rand()<self.scale_prob:
            min_scale, max_scale = (self.lminscale,self.lmaxscale) if min(h,w) < self.lhth else (self.hminscale,self.hmaxscale)
            scale_x = 2. ** np.random.uniform(min_scale, max_scale)
            scale_x = np.clip(scale_x, (cw+8) / float(w), None)
            scale_y = 1.
            
            if not self.scale_xonly:
                scale_y = scale_x
                scale_y = np.clip(scale_y, (ch+8) / float(h), None)            
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            disp = cv2.resize(disp, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_NEAREST) * scale_x
            if not append is None:
                new = {}
                for i in append:
                    new[i] = cv2.resize(append[i], None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_NEAREST)
                append = new
        else: # check if we need to resize to be able to crop 
            h,w = img1.shape[:2]
            clip_scale = (cw+8) / float(w)
            if clip_scale>1.: # do not scale the images
                scale_x = clip_scale
                scale_y = scale_x if not self.scale_xonly else 1.0 
                img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
                img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
                disp = cv2.resize(disp, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_NEAREST) * scale_x
                if not append is None:
                    new = {}
                    for i in append:
                        new[i] = cv2.resize(append[i], None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_NEAREST)
                    append = new
        return img1, img2, disp, append
    
    def _random_crop(self,img1,img2,disp,append=None):
        assert img1.shape[0] >= self.crop_size[0], str(img1.shape) +' '+ str(self.crop_size)
        assert img1.shape[1] >= self.crop_size[1], str(img1.shape) +' '+ str(self.crop_size)

        if img1.shape[0] == self.crop_size[0]:
            y0 = img1.shape[0]
        else:
            y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0])
        if img1.shape[1] == self.crop_size[1]:
            x0 = img1.shape[1]
        else:
            x0 = np.random.randint(0, img1.shape[1] - self.crop_size[1])
        
        y0 = np.clip(y0, 0, img1.shape[0] - self.crop_size[0])
        x0 = np.clip(x0, 0, img1.shape[1] - self.crop_size[1])

        img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        img2 = img2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        disp = disp[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        if not append is None:
            new = {}
            for i in append:
                new[i] = append[i][y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            append=new
        return img1,img2,disp, append
    
    def _HorizontalFlip(self,img1,img2,disp,append=None):
        if self.rng.uniform(0, 1, 1) > 0.5:
            img2_ =  np.copy(np.fliplr(img1))
            img1_ =  np.copy(np.fliplr(img2))
            disp =  np.copy(np.fliplr(disp))
            if not append is None:
                new = {}
                for i in append:
                    new[i] =  np.copy(np.fliplr(append[i]))
                append=new
        return img1_,img2_,disp, append

    def _VerticalFlip(self,img1,img2,disp,append=None):
        if self.rng.uniform(0, 1, 1) > 0.5:
            img1 =  np.copy(np.flipud(img1))
            img2 =  np.copy(np.flipud(img2))
            disp =  np.copy(np.flipud(disp))
            if not append is None:
                new = {}
                for i in append:
                    new[i] =  np.copy(np.flipud(append[i]))
                append=new
        return img1,img2,disp, append
    
    def _img_to_tensor(self,img1,img2,disp,append=None):
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1) / 255.
        img2 = torch.from_numpy(img2.copy()).permute(2, 0, 1) / 255.
        disp = torch.from_numpy(disp.copy())
        if not append is None:
            new = {}
            for i in append:
                new[i] = torch.from_numpy(append[i].copy())
            append=new
        return img1,img2,disp,append
    
    def _norm(self,img1,img2,disp,append=None):        
        img1 = (img1-in1k_mean)/in1k_std
        img2 = (img2-in1k_mean)/in1k_std
        return img1,img2,disp,append
    
    def __call__(self, sample):
        img1 = sample['left'] 
        img2 = sample['right'] 
        disp = sample['disp'] 
        append = sample['append']
        for t in self.transforms: # _img_to_tensor, _HorizontalFlip, _VerticalFlip, _random_crop, _random_scale_crop, _random_rotate, _resize__ScaleBase, _resize_SizeBase
            img1, img2, disp, append = t(img1, img2, disp, append = append)
        sample['left'] = img1 
        sample['right'] = img2 
        sample['disp'] = disp   
        sample['append'] = append
        return sample
    
