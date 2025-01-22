from torch.optim.lr_scheduler import CosineAnnealingLR

def train_simple_classifier(model, train_dataset, val_dataset, epochs=10, batch_size=32, learning_rate=0.001):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)  


    device = torch.device("cuda")
    model.to(device)

    for epoch in range(epochs):
        model.train() 
        running_loss = 0.0
        correct = 0
        total = 0

        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training", leave=False)
        for images, _, labels in train_loader_tqdm:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            train_loader_tqdm.set_postfix(loss=running_loss / len(train_loader_tqdm), accuracy=100 * correct / total)
        train_accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}, Train Accuracy: {train_accuracy}%")

        model.eval() 
        val_loss = 0.0
        correct = 0
        total = 0



        val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} Validation", leave=False)

        with torch.no_grad():  
            for images, _, labels, __ in val_loader_tqdm:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                val_loader_tqdm.set_postfix(loss=val_loss / len(val_loader_tqdm), accuracy=100 * correct / total)

        val_accuracy = 100 * correct / total
        print(f"Validation Loss: {val_loss / len(val_loader)}, Validation Accuracy: {val_accuracy}%")
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print(f"Learning rate for epoch {epoch+1}: {current_lr}")
    torch.save(model.state_dict(), 'simple2_model.pth')
    print("Model saved as simple2_model.pth")



def train_better_model(
    model,
    train_dataset,
    val_dataset,
    sampler,
    epochs=10,
    learning_rate=0.001,
    margin=2.0,
    synthetic_weight=0.5,
    use_tqdm=True
):
    """
    Обучает модель с использованием заданного сэмплера и комбинированной функции потерь.

    :param model: torch.nn.Module, модель для обучения.
    :param train_dataset: Dataset, тренировочный датасет.
    :param val_dataset: Dataset, валидационный датасет.
    :param sampler: Sampler, сэмплер для формирования батчей.
    :param epochs: int, количество эпох обучения.
    :param learning_rate: float, скорость обучения.
    :param margin: float, маржа для FeaturesLoss.
    :param synthetic_weight: float, вес синтетической части loss.
    :param use_tqdm: bool, использовать ли прогресс-бары.
    """
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=sampler)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    classification_criterion = nn.CrossEntropyLoss()
    features_criterion = FeaturesLoss(margin=margin)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_classification_loss = 0.0
        running_features_loss = 0.0
        correct = 0
        total = 0

        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training", leave=False, disable=not use_tqdm)
        for images, _, labels in train_loader_tqdm:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs, features = model(images) 
            classification_loss = classification_criterion(outputs, labels)
            features_loss = features_criterion(features, labels)

            loss = classification_loss + synthetic_weight * features_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_classification_loss += classification_loss.item()
            running_features_loss += features_loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            train_loader_tqdm.set_postfix(
                loss=running_loss / len(train_loader_tqdm),
                classification_loss=running_classification_loss / len(train_loader_tqdm),
                features_loss=running_features_loss / len(train_loader_tqdm),
                accuracy=100 * correct / total
            )

        train_accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}, Train Accuracy: {train_accuracy}%")

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} Validation", leave=False, disable=not use_tqdm)
        with torch.no_grad():
            for images, _, labels in val_loader_tqdm:
                images, labels = images.to(device), labels.to(device)

                outputs, features = model(images)
                classification_loss = classification_criterion(outputs, labels)
                features_loss = features_criterion(features, labels)
                loss = classification_loss + synthetic_weight * features_loss

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                val_loader_tqdm.set_postfix(
                    loss=val_loss / len(val_loader_tqdm),
                    accuracy=100 * correct / total
                )

        val_accuracy = 100 * correct / total
        print(f"Validation Loss: {val_loss / len(val_loader)}, Validation Accuracy: {val_accuracy}%")
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print(f"Learning rate for epoch {epoch+1}: {current_lr}")

    torch.save(model.state_dict(), 'better_model.pth')
    print("Model saved as better_model.pth")
