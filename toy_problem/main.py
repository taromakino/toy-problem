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
from utils.file import save_file


def make_data(args, task, eval_stage):
    batch_size = args.infer_batch_size if task == Task.CLASSIFY else args.batch_size
    data_train, data_val, data_test = MAKE_DATA[args.dataset](args.train_ratio, batch_size)
    if eval_stage is None:
        data_eval = None
    elif eval_stage == EvalStage.TRAIN:
        data_eval = data_train
    elif eval_stage == EvalStage.VAL:
        data_eval = data_val
    else:
        assert eval_stage == EvalStage.TEST
        data_eval = data_test
    return data_train, data_val, data_eval


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
        return vae.VAE(task, args.z_size, args.rank, args.h_sizes, args.prior_init_sd, args.y_mult,
            args.beta, args.reg_mult, args.lr, args.weight_decay, args.alpha, args.lr_infer, args.n_infer_steps)
    else:
        assert task == Task.CLASSIFY
        return vae.VAE.load_from_checkpoint(ckpt_fpath(args, Task.VAE), task=task, alpha=args.alpha, lr_infer=args.lr_infer,
            n_infer_steps=args.n_infer_steps)


def run_task(args, task, eval_stage):
    pl.seed_everything(args.seed)
    data_train, data_val, data_eval = make_data(args, task, eval_stage)
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
                    EarlyStopping(monitor='val_metric', mode='max', patience=int(args.early_stop_ratio * args.n_epochs)),
                    ModelCheckpoint(monitor='val_metric', mode='max', filename='best')],
                max_epochs=args.n_epochs)
            trainer.fit(model, data_train, data_val)
        else:
            trainer = pl.Trainer(logger=CSVLogger(os.path.join(args.dpath, task.value, eval_stage.value),
                name='', version=args.seed), max_epochs=1)
            trainer.test(model, data_eval)
    elif task == Task.VAE:
        trainer = pl.Trainer(
            logger=CSVLogger(os.path.join(args.dpath, task.value), name='', version=args.seed),
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=int(args.early_stop_ratio * args.n_epochs)),
                ModelCheckpoint(monitor='val_loss', filename='best')],
            max_epochs=args.n_epochs)
        trainer.fit(model, data_train, data_val)
        save_file(args, os.path.join(args.dpath, task.value, f'version_{args.seed}', 'args.pkl'))
    else:
        assert task == Task.CLASSIFY
        trainer = pl.Trainer(
            logger=CSVLogger(os.path.join(args.dpath, task.value, eval_stage.value), name='',
                version=args.seed),
            max_epochs=1,
            inference_mode=False)
        trainer.test(model, data_eval)


def main(args):
    if args.task == Task.ALL:
        # run_task(args, Task.VAE, None)
        # run_task(args, Task.CLASSIFY, EvalStage.VAL)
        # run_task(args, Task.CLASSIFY, EvalStage.TEST)
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
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--infer_batch_size', type=int, default=2048)
    parser.add_argument('--z_size', type=int, default=16)
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--h_sizes', nargs='+', type=int, default=[256, 256])
    parser.add_argument('--prior_init_sd', type=float, default=0.1)
    parser.add_argument('--y_mult', type=float, default=1)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--reg_mult', type=float, default=1e-5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--lr_infer', type=float, default=1)
    parser.add_argument('--n_infer_steps', type=int, default=200)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--early_stop_ratio', type=float, default=0.1)
    main(parser.parse_args())
