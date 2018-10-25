from optparse import OptionParser
# ensure .../chemprop is on PYTHONPATH
from chemprop.dio import load_split_data
from chemprop.run import train_model
from chemprop.run import load_chemprop_model_architecture
from chemprop.run import load_stored_weights
from chemprop.run import set_model_retrain
from chemprop.run import set_processor
from chemprop.run import valid_loss

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-t", "--data_train", dest="data_train_path")
    parser.add_option("-q", "--data_test", dest="data_test_path")
    parser.add_option("-m", "--model_path", dest="model_path")
    parser.add_option("-v", "--valid_split", dest="valid_split", default=.25)
    parser.add_option("-z", "--test_split", dest="test_split", default=.0)
    parser.add_option("-o", "--save_dir", dest="save_path")
    parser.add_option("-b", "--batch", dest="batch_size", default=50)
    parser.add_option("-w", "--hidden", dest="hidden_size", default=300)
    parser.add_option("-d", "--depth", dest="depth", default=3)
    parser.add_option("-e", "--epoch", dest="epoch", default=5)
    parser.add_option("-p", "--parameter_train", dest="param_train", default=False)
    parser.add_option("-r", "--dropout", dest="dropout", default=0)
    parser.add_option("-x", "--metric", dest="metric", default='rmse')
    parser.add_option("-s", "--seed", dest="seed", default=1)
    parser.add_option("-c", "--scale", dest="scale", default=True)
    parser.add_option("-i", "--initial_tasks", dest="initial_tasks", default=18)

    opts, args = parser.parse_args()
    batch_size = int(opts.batch_size)
    depth = int(opts.depth)
    hidden_size = int(opts.hidden_size)
    num_epoch = int(opts.epoch)
    dropout = float(opts.dropout)
    model = load_chemprop_model_architecture(num_tasks=initial_tasks)
    model = load_stored_weights(model, opts.model_path)
    model, loss_fn = set_processor(model)

    train, valid, _ = load_split_data(opts.data_test_path, 0.25, 0., opts.seed, opts.scale)
    num_tasks = len(train[0][1])
    model, optimizer, scheduler = set_model_retrain(model, hidden_size=hidden_size, num_tasks=num_tasks,
                                                    allow_parameters=opts.param_train)
    train_model(model, train, valid, optimizer=optimizer, scheduler=scheduler, loss_fn=loss_fn, batch_size=batch_size,
                num_epoch=num_epoch, metric=opts.metric, save_path=opts.save_path, verbose=False)
    ## Error on testing set
    test, _, _ = load_split_data(opts.data_test_path, 0., 0., opts.seed, opts.scale)  # abuse function to load test data
    model = load_chemprop_model_architecture(num_tasks=num_tasks)
    model = load_stored_weights(model, opts.save_path + "/model.best")
    print("test error: %.4f" % valid_loss(model, test, opts.metric))
