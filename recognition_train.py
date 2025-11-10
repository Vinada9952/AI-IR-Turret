import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# print("CUDA disponible :", torch.cuda.is_available())
# if torch.cuda.is_available():
#     print("Nom GPU :", torch.cuda.get_device_name(0))
#     x = torch.randn(3, 3).cuda()
#     print("Tenseur sur GPU :", x)
# else:
#     print("❌ Le GPU n'est pas accessible par PyTorch")

def main():
    prints = []

    # Paramètres
    torch.backends.cudnn.benchmark = True

    data_dir = './train_datas/'
    num_epochs = 75
    batch_size = 256
    num_workers = os.cpu_count()
    model_save_path = './models/recognition.pth'

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Dataset & DataLoader
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(
        dataset,
        shuffle=True,
        pin_memory=True,
        prefetch_factor=4,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True,
    )
    class_names = dataset.classes
    num_classes = len(class_names)

    print(f"Classes détectées : {class_names}")

    # Modèle
    model = models.resnet18( pretrained=True )
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Perte et optimiseur
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    # Entraînement
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to( device )
    print(f"--------\nAppareil utilisé : {device}\n--------")


    scaler = torch.cuda.amp.GradScaler()

    for epoch in tqdm( range(num_epochs) ):
        model.train()
        running_loss = 0.0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataset)
        printstr = f" Époque {epoch+1}/{num_epochs}, Perte: {epoch_loss:.4f}"
        print( printstr, end='\r' )
        prints.append( printstr )

    # Sauvegarde
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names
    }, model_save_path)


    nombres = ''
    for i in prints:
        nombres += i[len( i )-6:len( i )] + ' '
    liste_nombres = [float(x) for x in nombres.split()]

    print(f"✅ Entraînement terminé et modèle sauvegardé dans {model_save_path}")
    
    plt.bar(range(len(liste_nombres)), liste_nombres)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Diagramme de loss par epoch')
    plt.show()
    print( "fenêtre fermé" )
    exit()


# ✅ Obligatoire sous Windows
if __name__ == "__main__":
    main()
