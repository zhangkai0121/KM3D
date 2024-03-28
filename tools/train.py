import os
import torch
import torch.utils.data
import sys
sys.path.append('./src')
from lib.opts import opts
from lib.models.model import create_model, load_model, save_model
from lib.logger import Logger
from lib.datasets.dataset_factory import Dataset
from lib.trains.car_pose import CarPoseTrainer

def main(opt):
    torch.manual_seed(opt.seed)           # 设置随机数种子
    
    # 更新配置信息
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)    
    print(opt)

    # 实例化logger
    logger = Logger(opt)
    
    # 选择设备
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
    
    print('Creating model...')
 
    # 实例化模型
    model = create_model(opt.arch, opt.heads, opt.head_conv)
 
    # 实例化优化器
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    
    start_epoch = 0
    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)
    
    
    trainer = CarPoseTrainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device, opt.distribute)

    print('Setting up data...')
 
    # 构造验证集
    val_loader = torch.utils.data.DataLoader(
            Dataset(opt, 'val'), 
            batch_size=1, 
            shuffle=False,
            num_workers=1,
            pin_memory=True
    )

    # 仅测试时用
    if opt.test:
        _, preds = trainer.val(0, val_loader)
        val_loader.dataset.run_eval(preds, opt.save_dir)
        return
    
    # 构造训练集
    train_loader = torch.utils.data.DataLoader(
            Dataset(opt, 'train'), 
            batch_size=opt.batch_size, 
            shuffle=True,
            num_workers=opt.num_workers,
            pin_memory=True,
            drop_last=True)

    # 开始训练
    print('Starting training...')
    best = 1e10
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        mark = epoch if opt.save_all else 'last'
        log_dict_train, _ = trainer.train(epoch, train_loader)   # 训练
        
        # 打印log
        logger.write('epoch: {} |'.format(epoch))
        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))

        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), 
                                 epoch, model, optimizer)
            with torch.no_grad():
                log_dict_val, preds = trainer.val(epoch, val_loader)
            for k, v in log_dict_val.items():
                logger.scalar_summary('val_{}'.format(k), v, epoch)
                logger.write('{} {:8f} | '.format(k, v))
            if log_dict_val[opt.metric] < best:
                best = log_dict_val[opt.metric]
                save_model(os.path.join(opt.save_dir, 'model_best.pth'), 
                                     epoch, model)
        else:
            save_model(os.path.join(opt.save_dir, 'model_last.pth'), 
                                 epoch, model, optimizer)
        logger.write('\n')
        
        # 调整学习率
        if epoch in opt.lr_step:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                                            epoch, model, optimizer)
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    logger.close()

if __name__ == '__main__':
    opt = opts().parse()
    main(opt)
 