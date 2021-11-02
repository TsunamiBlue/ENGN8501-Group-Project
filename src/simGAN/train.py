import os
import time
import tqdm
import torch
import torch.utils.data as Data
import torchvision
from torch.autograd import Variable
from torchvision import transforms
from torch import nn

from src.simGAN.utils.buffer import ImagePool
from src.simGAN.model import Discriminator, Refiner
from src.simGAN.utils.external_func import get_accuracy, loop_iter, MyTimer
import src.simGAN.config as cfg


import torch.nn.functional as F

class Main(object):
    def __init__(self):
        # network
        self.G = None
        self.D = None
        self.refiner_optimizer = None
        self.discriminator_optimizer = None

        self.self_regularization_loss = None
        self.local_adversarial_loss = None
        self.delta = None
        self.my_timer = MyTimer()
        # data
        self.fake_images_loader = None
        self.real_images_loader = None
        self.fake_images_iter = None
        self.real_images_iter = None
        self.current_step = 0

    def build_network(self):
        print('Building network ...')
        self.G = Refiner(4, cfg.img_channels, nb_features=64)
        self.D = Discriminator(input_features=cfg.img_channels)

        if cfg.cuda_use:
            self.G.cuda(cfg.cuda_num)
            self.D.cuda(cfg.cuda_num)

        self.refiner_optimizer = torch.optim.SGD(self.G.parameters(), lr=cfg.init_lr, momentum=0.9)
        self.discriminator_optimizer = torch.optim.SGD(self.D.parameters(), lr=cfg.init_lr, momentum=0.9)

        # self.refiner_scheduler = StepLR(self.refiner_optimizer, step_size=2000, gamma=0.1)
        # self.discriminator_scheduler = StepLR(self.discriminator_optimizer, step_size=2000, gamma=0.1)

        self.self_regularization_loss = nn.L1Loss(size_average=True)
        self.local_adversarial_loss = nn.CrossEntropyLoss(size_average=True)  # LocalAdversarialLoss()
        self.delta = cfg.delta

    def load_previous_mode(self):
        if not os.path.isdir(cfg.save_path):
            os.mkdir(cfg.save_path)
        ckpts = os.listdir(cfg.save_path)
        refiner_ckpts = [ckpt for ckpt in ckpts if 'R_' in ckpt]
        refiner_ckpts.sort(key=lambda x: int(x[2:-4]), reverse=True)
        discriminator_ckpts = [ckpt for ckpt in ckpts if 'D_' in ckpt]
        discriminator_ckpts.sort(key=lambda x: int(x[2:-4]), reverse=True)

        if len(refiner_ckpts) == 0 or len(discriminator_ckpts) == 0 or not os.path.isfile(
                os.path.join(cfg.save_path, cfg.optimizer_path)):
            return True

        optimizer_status = torch.load(os.path.join(cfg.save_path, cfg.optimizer_path))
        self.refiner_optimizer.load_state_dict(optimizer_status['optR'])
        self.discriminator_optimizer.load_state_dict(optimizer_status['optD'])
        self.current_step = optimizer_status['step']

        # Load pretrained model
        print('Loading previous model {} and {} ...'.format(refiner_ckpts[0], discriminator_ckpts[0]))
        self.D.load_state_dict(torch.load(os.path.join(cfg.save_path, discriminator_ckpts[0])))
        self.G.load_state_dict(torch.load(os.path.join(cfg.save_path, refiner_ckpts[0])))

        return False

    def get_data_loaders(self):
        print('Creating dataloaders ...')
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((cfg.img_width, cfg.img_height), interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        fake_folder = torchvision.datasets.ImageFolder(root=cfg.syn_path, transform=transform)
        real_folder = torchvision.datasets.ImageFolder(root=cfg.real_path, transform=transform)

        self.fake_images_loader = Data.DataLoader(fake_folder, batch_size=cfg.batch_size, shuffle=True,
                                                  pin_memory=True, drop_last=True, num_workers=3)
        self.real_images_loader = Data.DataLoader(real_folder, batch_size=cfg.batch_size, shuffle=True,
                                                  pin_memory=True, drop_last=True, num_workers=3)
        self.fake_images_iter = loop_iter(self.fake_images_loader)
        self.real_images_iter = loop_iter(self.real_images_loader)

    def pretrain_generator(self):
        # we first train the Rθ network with just self-regularization loss for 1,000 steps
        print('pre-training the refiner network %d times...' % cfg.g_pretrain)
        self.G.train()
        # fake_iter = iter(self.fake_images_loader)
        for step in range(cfg.g_pretrain):
            fake_images, _ = next(self.fake_images_iter)
            fake_images = Variable(fake_images).cuda(cfg.cuda_num)
            refined_images = self.G(fake_images)
            # regularization loss
            reg_loss = self.self_regularization_loss(refined_images, fake_images)
            # reg_loss = torch.mul(reg_loss, self.delta)
            # update model
            self.refiner_optimizer.zero_grad()
            reg_loss.backward()
            self.refiner_optimizer.step()
            # save
            if (step % cfg.r_pre_per == 0) or (step == cfg.g_pretrain - 1):
                print('------Step[%d/%d]------' % (step, cfg.g_pretrain))
                print('# Refiner: loss: %.4f' % (reg_loss.data[0]))
                torch.save(self.G.state_dict(), os.path.join(cfg.save_path, 'R_0.pkl'))

    def pretrain_discrimintor(self):
        # and Dφ for 200 steps (one mini-batch for refined images, another for real)
        print('pre-training the discriminator network %d times...' % cfg.d_pretrain)
        self.D.train()
        self.G.eval()
        for step in range(cfg.d_pretrain):
            # for real images
            real_images, _ = next(self.real_images_iter)
            real_images = Variable(real_images).cuda(cfg.cuda_num)
            real_predictions = self.D(real_images)
            real_labels = Variable(torch.ones(real_predictions.size(0)).type(torch.LongTensor)).cuda(cfg.cuda_num)
            acc_real = get_accuracy(real_predictions, 'real')
            loss_real = self.local_adversarial_loss(real_predictions, real_labels)

            # for fake images
            fake_images, _ = next(self.fake_images_iter)
            fake_images = Variable(fake_images).cuda(cfg.cuda_num)
            refined_images = self.G(fake_images)
            refined_images = refined_images.detach()
            # refined_images = fake_images
            refined_predictions = self.D(refined_images)
            refined_labels = Variable(torch.zeros(refined_predictions.size(0)).type(torch.LongTensor)).cuda(
                cfg.cuda_num)
            acc_ref = get_accuracy(refined_predictions, 'refine')
            loss_ref = self.local_adversarial_loss(refined_predictions, refined_labels)

            self.discriminator_optimizer.zero_grad()
            (loss_real + loss_ref).backward()
            self.discriminator_optimizer.step()

            if step % cfg.d_pre_per == 0 or (step == cfg.d_pretrain - 1):

                real_p = F.softmax(real_predictions, dim=-1).cpu().data.numpy()

                ref_p = F.softmax(refined_predictions, dim=-1).cpu().data.numpy()


                d_loss = (loss_real + loss_ref) / 2

                print('------Step[%d/%d]------' % (step, cfg.d_pretrain))
                print('# Discriminator: loss:%f  accuracy_real:%.2f accuracy_ref:%.2f'
                      % (d_loss.data[0], acc_real, acc_ref))

        print('Save D_pre to models/D_0.pkl')
        torch.save(self.D.state_dict(), os.path.join(cfg.save_path, 'D_0.pkl'))

        # self.discriminator_optimizer = torch.optim.SGD(self.D.parameters(), lr=cfg.init_lr)


    def train(self):
        print('Start Formal Training ...')
        image_pool = ImagePool(cfg.buffer_size)
        assert self.current_step < cfg.train_steps, 'Target step is smaller than current step!'
        step_timer = time.time()

        for step in range((self.current_step + 1), cfg.train_steps):
            self.current_step = step
            self.D.eval()
            self.D.train_mode(False)
            self.G.train()

            for index in tqdm.tqdm(range(cfg.k_g)):
                self.my_timer.track()
                fake_images, _ = next(self.fake_images_iter)
                fake_images = Variable(fake_images).cuda(cfg.cuda_num)
                self.my_timer.add_value('Read Fake Images')
                # forward #1
                self.my_timer.track()
                refined_images = self.G(fake_images)
                self.my_timer.add_value('Refine Fake Images')
                # forward #2
                self.my_timer.track()
                refined_predictions = self.D(refined_images).view(-1, 2)
                self.my_timer.add_value('Predict Fake Images')
                # calculate loss
                self.my_timer.track()
                refined_labels = Variable(torch.ones(refined_predictions.size(0)).type(torch.LongTensor)).cuda(
                    cfg.cuda_num)
                reg_loss = self.self_regularization_loss(refined_images, fake_images)
                reg_loss = torch.mul(reg_loss, self.delta)
                adv_loss = self.local_adversarial_loss(refined_predictions, refined_labels)

                refine_loss = reg_loss + adv_loss
                self.my_timer.add_value('Get Refine Loss')


                ref_p = F.softmax(refined_predictions, dim=-1).cpu().data.numpy()

                # backward
                self.my_timer.track()
                # has made some changes
                self.refiner_optimizer.zero_grad()
                # self.refiner_optimizer.zero_grad()
                refine_loss.backward()
                self.refiner_optimizer.step()
                self.my_timer.add_value('Backward Refine Loss')

            # ========= train the D =========
            self.G.eval()
            self.D.train_mode(True)
            self.D.train()

            for index in range(cfg.k_d):
                # get images
                self.my_timer.track()
                fake_images, _ = next(self.fake_images_iter)
                fake_images = Variable(fake_images).cuda(cfg.cuda_num)
                real_images, _ = next(self.real_images_iter)
                real_images = Variable(real_images).cuda(cfg.cuda_num)
                self.my_timer.add_value('Read All Images')
                # generate refined images
                self.my_timer.track()
                refined_images = self.G(fake_images)
                self.my_timer.add_value('Refine Fake Images')
                # use a history of refined images
                self.my_timer.track()
                # refined_images = refined_images.detach()
                images_diff = torch.mean(torch.abs(refined_images - fake_images)).cpu().data.numpy()

                refined_images = refined_images.detach().cpu()
                refined_images = image_pool.query(refined_images)
                refined_images = refined_images.cuda(cfg.cuda_num)

                self.my_timer.add_value('Get History Images')
                # predict images
                self.my_timer.track()
                real_predictions = self.D(real_images).view(-1, 2)
                refined_predictions = self.D(refined_images).view(-1, 2)
                self.my_timer.add_value('Predict All Images')
                # get all loss
                self.my_timer.track()

                real_labels = Variable(torch.ones(real_predictions.size(0)).type(torch.LongTensor)).cuda(
                    cfg.cuda_num)
                refined_labels = Variable(torch.zeros(refined_predictions.size(0)).type(torch.LongTensor)).cuda(
                    cfg.cuda_num)

                pred_loss_real = self.local_adversarial_loss(real_predictions, real_labels)
                acc_real = get_accuracy(real_predictions, 'real')
                pred_loss_ref = self.local_adversarial_loss(refined_predictions, refined_labels)
                acc_ref = get_accuracy(refined_predictions, 'refine')
                self.my_timer.add_value('Get Combine Loss')

                self.my_timer.track()
                # backward discriminator loss
                self.discriminator_optimizer.zero_grad()
                d_loss = (pred_loss_ref + pred_loss_real) / 2.
                d_loss.backward()
                self.discriminator_optimizer.step()

                self.my_timer.add_value('Backward Combine Loss')

            # udpate learning rate
            # self.refiner_scheduler.step()
            # self.discriminator_scheduler.step()

            if step % cfg.f_per == 0:
                print('------Step[%d/%d]------Time Cost: %.2f seconds' % (
                step, cfg.train_steps, time.time() - step_timer))
                print('# Refiner: loss:%.4f reg_loss:%.4f, adv_loss:%.4f' % (
                    refine_loss.data[0], reg_loss.data[0], adv_loss.data[0]))
                print('# Discrimintor: loss:%.4f real:%.4f(%.2f) refined:%.4f(%.2f)'
                      % (d_loss.data[0], pred_loss_real.data[0], acc_real, pred_loss_ref.data[0], acc_ref))

                real_p = F.softmax(real_predictions, dim=-1).cpu().data.numpy()

                ref_p = F.softmax(refined_predictions, dim=-1).cpu().data.numpy()

                time_dict = self.my_timer.get_all_time()
                step_timer = time.time()

            if step % cfg.save_per == 0 and step > 0:
                print('Save two model dict.')
                torch.save(self.D.state_dict(), os.path.join(cfg.save_path, cfg.D_path % step))
                torch.save(self.G.state_dict(), os.path.join(cfg.save_path, cfg.R_path % step))
                state = {
                    'step': step,
                    'optD': self.discriminator_optimizer.state_dict(),
                    'optR': self.refiner_optimizer.state_dict()
                }
                torch.save(state, os.path.join(cfg.save_path, cfg.optimizer_path))


if __name__ == '__main__':
    obj = Main()
    obj.build_network()
    obj.get_data_loaders()

    if obj.load_previous_mode():
        obj.pretrain_generator()
        obj.pretrain_discrimintor()

    obj.train()
