"""Training utility shared by the GRU, BiLSTM and Transformer baselines."""

import torch


def train_model_with_flag_masking(
    model,
    train_dataloader,
    valid_dataloader,
    criterion,
    optimizer,
    scheduler,
    EPOCHS,
    random_seed=42,
    patience=50,
):
    """Train a baseline model while excluding interpolated targets."""
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    best_validation_loss = float("inf")
    no_improve = 0
    last_lr = optimizer.param_groups[0]["lr"]

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss = 0
        count_actual_values_train = 0
        for train_inputs, train_targets, train_flags in train_dataloader:
            optimizer.zero_grad()
            train_outputs = model(train_inputs)
            if isinstance(train_outputs, tuple):
                train_outputs = train_outputs[0]
            mask = train_flags == 0
            train_outputs_masked = torch.where(
                mask, train_outputs, torch.tensor(0.0, device=train_outputs.device)
            )
            train_targets_masked = torch.where(
                mask, train_targets, torch.tensor(0.0, device=train_targets.device)
            )

            if mask.sum() > 0:
                train_loss = criterion(train_outputs_masked, train_targets_masked)
                train_loss.backward()
                optimizer.step()
                total_train_loss += train_loss.item()
                count_actual_values_train += 1

        avg_train_loss = (
            total_train_loss / count_actual_values_train
            if count_actual_values_train > 0
            else float("inf")
        )

        model.eval()
        validation_loss = 0
        count_actual_values = 0
        with torch.no_grad():
            for valid_inputs, valid_targets, valid_flags in valid_dataloader:
                valid_outputs = model(valid_inputs)
                if isinstance(valid_outputs, tuple):
                    valid_outputs = valid_outputs[0]
                mask = valid_flags == 0
                valid_outputs_masked = torch.where(
                    mask, valid_outputs, torch.tensor(0.0, device=valid_outputs.device)
                )
                valid_targets_masked = torch.where(
                    mask, valid_targets, torch.tensor(0.0, device=valid_targets.device)
                )

                if mask.sum() > 0:
                    valid_loss = criterion(valid_outputs_masked, valid_targets_masked)
                    validation_loss += valid_loss.item()
                    count_actual_values += 1

        if count_actual_values > 0:
            validation_loss /= count_actual_values

        current_lr = optimizer.param_groups[0]["lr"]
        if epoch % 5 == 0 or current_lr != last_lr:
            print(
                f"Epoch {epoch}/{EPOCHS}, Training Loss: {avg_train_loss:.4f}, "
                f"Validation Loss: {validation_loss:.4f}, LR: {current_lr:.6f}"
            )

        if current_lr != last_lr:
            print(f"Learning rate changed to {current_lr:.6f} at epoch {epoch}")
            last_lr = current_lr

        scheduler.step(validation_loss)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(
                f"Early stopping at epoch {epoch}, "
                f"best validation loss: {best_validation_loss:.4f}"
            )
            return model

    return model
