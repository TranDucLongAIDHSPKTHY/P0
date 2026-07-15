import torch
from time import time
from tqdm import tqdm

import utility.utility_train.batch_test as batch_test
import utility.utility_function.tools as tools


def universal_trainer(model, args, config, dataset, device, logger):
    model.to(device)
    Optim = torch.optim.Adam(model.parameters(), lr=float(config["learn_rate"]))
    logger.info("Optimizer Initialized: Adam lr=%s", config["learn_rate"])

    best_results = {
        "count": 0, "epoch": 0,
        "recall": [0.0 for _ in eval(config["top_K"])],
        "ndcg": [0.0 for _ in eval(config["top_K"])],
        "stop": 0,
    }
    for epoch in range(int(config["training_epochs"])):
        dataset.current_epoch = epoch + 1
        logger.info("Start Epoch: %d", epoch + 1)
        start_time = time()
        model.train()

        sample_data = dataset.sample_data_to_train_all()
        users = torch.Tensor(sample_data[:, 0]).long().to(device)
        pos_items = torch.Tensor(sample_data[:, 1]).long().to(device)
        neg_items = torch.Tensor(sample_data[:, 2]).long().to(device)
        users, pos_items, neg_items = tools.shuffle(users, pos_items, neg_items)
        num_batch = len(users) // int(config["batch_size"]) + 1
        total_loss_list = []

        batches = tools.mini_batch(users, pos_items, neg_items, batch_size=int(config["batch_size"]))
        for batch_i, (batch_users, batch_positive, batch_negative) in tqdm(
            enumerate(batches), desc="Training epoch " + str(epoch + 1), total=int(num_batch)
        ):
            loss_list = model(batch_users, batch_positive, batch_negative)
            if batch_i == 0:
                assert len(loss_list) >= 1
                total_loss_list = [0.0] * len(loss_list)
            total_loss = 0.0
            for index, loss in enumerate(loss_list):
                total_loss += loss
                total_loss_list[index] += loss.item()
            Optim.zero_grad()
            total_loss.backward()
            Optim.step()

        elapsed = time() - start_time
        loss_strs = str(round(sum(total_loss_list) / num_batch, 6)) + " = " + " + ".join(
            str(round(value / num_batch, 6)) for value in total_loss_list
        )
        logger.info("Train Loss: epoch=%d loss=%s", epoch + 1, loss_strs)
        logger.info("End Epoch: %d training_time=%.3f", epoch + 1, elapsed)
        _, best_results = batch_test.general_test(
            dataset, model, device, config, epoch, best_results, Optim, logger
        )
        if best_results["stop"] > 0:
            break

    logger.info(
        "Training loop completed: best_epoch=%d best_recall=%s best_ndcg=%s",
        best_results["epoch"], best_results["recall"], best_results["ndcg"],
    )
