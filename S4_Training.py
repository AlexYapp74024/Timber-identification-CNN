from collections import Counter
from tqdm import tqdm
from time import time
import torch
from torch import nn, Tensor
from S3_intermediateDataset import build_intermediate_dataset_if_not_exists, intermediate_dataset
from S2_TimberDataset import build_dataloader
from S1_CNN_Model import build_model

if __name__ == '__main__':
    img_size = (320,320)
    train_loader, val_loader = build_dataloader(
        # train_ratio= 0.005,
        img_size=img_size,
        batch_size=16,
    )
    
    build_intermediate_dataset_if_not_exists(lambda x:x, "train", train_loader)
    build_intermediate_dataset_if_not_exists(lambda x:x, "val", val_loader)
    
    train_loader = intermediate_dataset("train") 
    val_loader = intermediate_dataset("val") 

    model = build_model(img_size=img_size)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    accuracies = []

    n_epoch = 40
    timer = time()
    pbar_0 = tqdm(range(n_epoch), position=0, ncols=100)
    pbar_0.set_description(f"epoch 1/{n_epoch}")
    img = None
    for epoch in pbar_0:
        pbar_1 = tqdm(enumerate(train_loader), total=len(train_loader), position=1, ncols=100, leave=False)
        for i, (images, labels) in pbar_1:
            # # Reshape
            images = images.reshape(images.shape[1:])
            labels = labels.reshape(labels.shape[1:])
            # Forward
            out = model.forward(images)
            loss = criterion.forward(out, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            optimizer.step()

            if i % 10 == 0:
                pbar_1.set_description(f"  loss = {loss.item():.4f} ({(time() - timer)*1000:.4f} ms)")
                timer = time()
        
        n_correct = 0
        n_samples = 0
        pbar_0.set_description(f"epoch {epoch+1}/{n_epoch}, Validating . . .")

        with torch.no_grad(): 
            with tqdm(bar_format='{desc}{postfix}', position=1, leave=False) as val_desc:
                tally = Counter()
                for images, labels in tqdm(val_loader, position=2, ncols=100, leave=False):
                    # # Reshape
                    images = images.reshape(images.shape[1:])
                    labels = labels.reshape(labels.shape[1:])

                    x = model.forward(images)

                    _, predictions = torch.max(x,1)
                    tally += Counter(predictions.tolist())
                    n_samples += labels.shape[0]
                    n_correct += (predictions == labels).sum().item()
                    tally_desc = ' '.join([f"{n}:{c}" for n,c in tally.most_common()])[:80] + "..."
                    val_desc.set_description(f"{n_correct}/{n_samples} correct")
                    val_desc.set_postfix_str(tally_desc)
                
                accuracy = f"{n_correct/n_samples * 100:.2f}%"
                pbar_0.set_description(f"epoch {epoch+2}/{n_epoch}, accuracy: {accuracy}")
                
                if len(accuracies) >= 3 and accuracy > max(accuracies):
                    torch.save(model,f"model_{epoch}.pt")

                accuracies.append(accuracy)

    torch.save(model,"model.pt")
    
    print(accuracies)