import models.erm as erm
import models.vae as vae
import os
import pytorch_lightning as pl
import plot_reconstruction
from argparse import ArgumentParser
from data import MAKE_DATA
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from utils.enums import Task, EvalStage


def make_data(args, task, eval_stage):
    data_train, data_val, data_test = MAKE_DATA[args.dataset](args.train_ratio, args.batch_size, args.eval_batch_size,
        args.n_eval_examples if task == Task.VAE else None)
    if eval_stage is None:
        data_eval = None
    elif eval_stage == EvalStage.TRAIN:
        data_eval = data_train
    elif eval_stage == EvalStage.VAL:
        data_eval = data_val
    else:
        assert eval_stage == EvalStage.TEST
        data_eval = data_test
    return data_train, data_val, data_test, data_eval


def ckpt_fpath(args, task):
    return os.path.join(args.dpath, task.value, f'version_{args.seed}', 'checkpoints', 'best.ckpt')


def make_model(args, task, is_train):
    if task == Task.ERM_X:
        if is_train:
            return erm.ERM_X(args.h_sizes, args.lr, args.weight_decay)
        else:
            return erm.ERM_X.load_from_checkpoint(ckpt_fpath(args, task))
    elif task == Task.ERM_C:
        if is_train:
            return erm.ERM_C(args.h_sizes, args.lr, args.weight_decay)
        else:
            return erm.ERM_C.load_from_checkpoint(ckpt_fpath(args, task))
    elif task == Task.ERM_S:
        if is_train:
            return erm.ERM_S(args.h_sizes, args.lr, args.weight_decay)
        else:
            return erm.ERM_S.load_from_checkpoint(ckpt_fpath(args, task))
    elif task == Task.VAE:
        return vae.VAE(args.task, args.z_size, args.rank, args.h_sizes, args.y_mult, args.beta, args.reg_mult, args.init_sd,
            args.n_samples, args.lr, args.weight_decay, args.lr_infer, args.n_infer_steps)
    else:
        assert task == Task.CLASSIFY
        return vae.VAE.load_from_checkpoint(ckpt_fpath(args, Task.VAE), task=args.task, lr_infer=args.lr_infer,
            n_infer_steps=args.n_infer_steps)


def run_task(args, task, eval_stage):
    pl.seed_everything(args.seed)
    data_train, data_val, data_test, data_eval = make_data(args, task, eval_stage)
    is_train = eval_stage is None
    model = make_model(args, task, is_train)
    if task in [
        Task.ERM_X,
        Task.ERM_C,
        Task.ERM_S
    ]:
        if is_train:
            trainer = pl.Trainer(
                logger=CSVLogger(os.path.join(args.dpath, task.value), name='', version=args.seed),
                callbacks=[
                    EarlyStopping(monitor='val_acc', mode='max', patience=int(args.early_stop_ratio * args.n_epochs)),
                    ModelCheckpoint(monitor='val_acc', mode='max', filename='best')],
                max_epochs=args.n_epochs,
                deterministic=True)
            trainer.fit(model, data_train, data_val)
        else:
            trainer = pl.Trainer(
                logger=CSVLogger(os.path.join(args.dpath, task.value, eval_stage.value), name='', version=args.seed),
                max_epochs=1,
                deterministic=True)
            trainer.test(model, data_eval)
    elif task == Task.VAE:
        trainer = pl.Trainer(
            logger=CSVLogger(os.path.join(args.dpath, task.value), name='', version=args.seed),
            callbacks=[
                EarlyStopping(monitor='val_acc', mode='max', patience=int(args.early_stop_ratio * args.n_epochs)),
                ModelCheckpoint(monitor='val_acc', mode='max', filename='best')],
            max_epochs=args.n_epochs,
            check_val_every_n_epoch=args.check_val_every_n_epoch,
            deterministic=True,
            inference_mode=False)
        trainer.fit(model, data_train, [data_val, data_test])
    else:
        assert task == Task.CLASSIFY
        trainer = pl.Trainer(
            logger=CSVLogger(os.path.join(args.dpath, task.value, eval_stage.value), name='', version=args.seed),
            max_epochs=1,
            deterministic=True,
            inference_mode=False)
        trainer.test(model, data_eval)


def main(args):
    if args.task == Task.ALL:
        run_task(args, Task.VAE, None)
        run_task(args, Task.CLASSIFY, EvalStage.VAL)
        run_task(args, Task.CLASSIFY, EvalStage.TEST)
        plot_reconstruction.main(args)
    else:
        run_task(args, args.task, args.eval_stage)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dpath', type=str, default='results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--task', type=Task, choices=list(Task), required=True)
    parser.add_argument('--eval_stage', type=EvalStage, choices=list(EvalStage))
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--eval_batch_size', type=int, default=1024)
    parser.add_argument('--n_eval_examples', type=int, default=1024)
    parser.add_argument('--z_size', type=int, default=32)
    parser.add_argument('--rank', type=int, default=32)
    parser.add_argument('--h_sizes', nargs='+', type=int, default=[256, 256])
    parser.add_argument('--y_mult', type=float, default=1)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--reg_mult', type=float, default=1e-5)
    parser.add_argument('--init_sd', type=float, default=0.1)
    parser.add_argument('--n_samples', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--lr_infer', type=float, default=1)
    parser.add_argument('--n_infer_steps', type=int, default=200)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=5)
    parser.add_argument('--early_stop_ratio', type=float, default=0.1)
    main(parser.parse_args())
