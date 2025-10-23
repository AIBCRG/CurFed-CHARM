import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import glob
import logging
import sys
from collections import OrderedDict
import copy
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('sequential_federated_training.log')
    ]
)
logger = logging.getLogger(__name__)


class ClientDataset(Dataset):
    def __init__(self, client_dir, sequence_length=8, transform=None, available_classes=None):
        self.client_dir = client_dir
        self.sequence_length = sequence_length
        self.transform = transform
        self.sequences = []
        self.labels = []

        self.action_classes = {
            'Bend Down': 0, 'Getting Up': 1, 'Jumping': 2, 'Left': 3,
            'Move Back': 4, 'Move Forward Fast': 5, 'Move Forward Slow': 6,
            'Right': 7, 'Stop': 8, 'Turn Around': 9, 'Walking': 10
        }
        
        # If available_classes is provided, filter the action_classes dictionary
        self.available_classes = available_classes
        
        logger.info(f"Creating sequences for {os.path.basename(client_dir)}")
        logger.info(f"Available classes: {available_classes if available_classes else 'All classes'}")
        self._create_sequences()

    def _create_sequences(self):
        for action_class, class_idx in self.action_classes.items():
            # Skip classes that aren't available for this client
            if self.available_classes is not None and class_idx not in self.available_classes:
                continue
                
            action_path = os.path.join(self.client_dir, action_class)

            if not os.path.exists(action_path):
                continue

            for sequence_folder in os.listdir(action_path):
                sequence_path = os.path.join(action_path, sequence_folder)
                if not os.path.isdir(sequence_path):
                    continue

                frames = []
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    frames.extend(glob.glob(os.path.join(sequence_path, ext)))
                frames = sorted(frames)

                if len(frames) < self.sequence_length:
                    continue

                for i in range(0, len(frames) - self.sequence_length + 1):
                    self.sequences.append(frames[i:i + self.sequence_length])
                    self.labels.append(self.action_classes[action_class])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        frames = []

        for frame_path in sequence:
            image = Image.open(frame_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            frames.append(image)

        frames = torch.stack(frames)
        frames = frames.permute(1, 0, 2, 3)  # [C, T, H, W]

        return frames, self.labels[idx]


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(AttentionBlock, self).__init__()
        self.spatial_avg_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.channel_fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )
        
        # Temporal attention
        self.temporal_avg_pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.temporal_fc = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction_ratio, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, t, h, w = x.size()
        
        # Channel attention
        y_channel = self.spatial_avg_pool(x).view(batch_size, channels, t)
        y_channel = y_channel.mean(dim=2)
        y_channel = self.channel_fc(y_channel).view(batch_size, channels, 1, 1, 1)
        
        # Temporal attention
        y_temporal = self.temporal_avg_pool(x)
        y_temporal = self.temporal_fc(y_temporal)
        
        # Apply attention
        x = x * y_channel * y_temporal
        return x


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # Add attention mechanism
        self.attention = AttentionBlock(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.attention(out)  # Apply attention
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class EnhancedActionRecognitionModel(nn.Module):
    def __init__(self, num_classes=11):
        super(EnhancedActionRecognitionModel, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=7, stride=(1, 2, 2), padding=3),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )

        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=(1, 2, 2))
        
        # Temporal attention after feature extraction
        self.temporal_attention = nn.Sequential(
            nn.Conv3d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResidualBlock3D(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock3D(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        
        # Apply temporal attention
        attn = self.temporal_attention(x)
        x = x * attn
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def train_client(model, train_loader, criterion, optimizer, device, epochs=10):
    model.train()
    history = {'loss': [], 'accuracy': []}

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)

        logger.info(f'Epoch {epoch + 1}/{epochs}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.2f}%')

    return model.state_dict(), history


def weighted_average_models(models_dict_list, client_weights):
    """
    Average model parameters weighted by client contributions.
    
    Args:
        models_dict_list: List of state dictionaries from client models
        client_weights: List of weights for each client (could be based on data size, 
                       performance, or class representation)
        
    Returns:
        Weighted averaged state dictionary
    """
    # Normalize weights
    total_weight = sum(client_weights)
    normalized_weights = [w / total_weight for w in client_weights]
    
    logger.info(f"Normalized client weights: {normalized_weights}")
    
    averaged_dict = OrderedDict()
    for i, client_dict in enumerate(models_dict_list):
        client_weight = normalized_weights[i]
        for key in client_dict.keys():
            if key not in averaged_dict:
                averaged_dict[key] = client_dict[key].float() * client_weight
            else:
                averaged_dict[key] += client_dict[key].float() * client_weight
                
    # Convert back to original dtype
    for key in averaged_dict:
        averaged_dict[key] = averaged_dict[key].to(models_dict_list[0][key].dtype)
                
    return averaged_dict


def evaluate_model(model, test_loader, criterion, device, class_names=None):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    # Store per-class accuracies
    class_correct = {}
    class_total = {}
    
    if class_names:
        for class_idx in range(len(class_names)):
            class_correct[class_idx] = 0
            class_total[class_idx] = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Track per-class performance
            if class_names:
                for i, label in enumerate(labels):
                    class_idx = label.item()
                    class_total[class_idx] += 1
                    if predicted[i] == label:
                        class_correct[class_idx] += 1

    test_loss = running_loss / len(test_loader)
    test_acc = 100. * correct / total
    
    # Calculate per-class accuracies
    per_class_acc = {}
    if class_names:
        for class_idx in range(len(class_names)):
            if class_total[class_idx] > 0:
                per_class_acc[class_idx] = 100. * class_correct[class_idx] / class_total[class_idx]
            else:
                per_class_acc[class_idx] = 0.0

    return test_loss, test_acc, all_preds, all_labels, per_class_acc


def plot_confusion_matrix(true_labels, pred_labels, classes, save_path):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_training_curves(client_histories, save_path):
    plt.figure(figsize=(15, 5))

    # Plot losses
    plt.subplot(1, 2, 1)
    for client_id, history in client_histories.items():
        plt.plot(history['loss'], label=f'Client {client_id}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss per Client')

    # Plot accuracies
    plt.subplot(1, 2, 2)
    for client_id, history in client_histories.items():
        plt.plot(history['accuracy'], label=f'Client {client_id}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training Accuracy per Client')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_class_distribution(client_classes, class_names, save_path):
    """Plot the class distribution across clients"""
    plt.figure(figsize=(15, 8))
    
    # Create a matrix to represent class availability per client
    availability_matrix = np.zeros((len(client_classes), len(class_names)))
    
    for i, (client_id, available_classes) in enumerate(client_classes.items()):
        for class_idx in available_classes:
            availability_matrix[i, class_idx] = 1
    
    # Create heatmap - use fmt='.0f' instead of 'd' to handle float values as integers
    sns.heatmap(availability_matrix, annot=True, fmt='.0f', cmap='YlGnBu',
               xticklabels=class_names, yticklabels=[f'Client {cid}' for cid in client_classes.keys()])
    plt.title('Class Distribution Across Clients')
    plt.xlabel('Action Classes')
    plt.ylabel('Clients')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# New specialized federated learning evaluation metrics

def calculate_client_convergence_rate(client_histories):
    """
    Client Convergence Rate (CCR): Measures how quickly clients reach stable performance.
    
    Args:
        client_histories: Dictionary of client training histories
        
    Returns:
        Dictionary of client convergence rates
    """
    ccr_results = {}
    
    for client_id, history in client_histories.items():
        # Calculate moving average of last 3 epochs to determine stability
        losses = history['loss']
        accuracies = history['accuracy']
        
        # Define convergence as having less than 1% change in both loss and accuracy
        # for 3 consecutive epochs
        convergence_epoch = None
        for i in range(3, len(losses)):
            loss_change = abs(losses[i] - losses[i-3]) / max(losses[i-3], 1e-6)
            acc_change = abs(accuracies[i] - accuracies[i-3]) / max(accuracies[i-3], 1e-6)
            
            if loss_change < 0.01 and acc_change < 0.01:
                convergence_epoch = i
                break
        
        # Calculate convergence rate (lower is better, faster convergence)
        if convergence_epoch is not None:
            ccr = convergence_epoch / len(losses)
        else:
            ccr = 1.0  # Didn't converge
            
        ccr_results[client_id] = ccr
    
    return ccr_results


def calculate_knowledge_transfer_efficiency(initial_accuracies, final_accuracies, client_train_classes):
    """
    Knowledge Transfer Efficiency (KTE): Measures how well knowledge transfers to classes 
    not present in client's local training data.
    
    Args:
        initial_accuracies: Dictionary of client accuracies before training (per class)
        final_accuracies: Dictionary of client accuracies after training (per class)
        client_train_classes: Dictionary of classes available to each client
        
    Returns:
        Dictionary of client KTE values
    """
    kte_results = {}
    
    for client_id, final_acc_dict in final_accuracies.items():
        # Get classes that this client didn't train on
        all_classes = set(final_acc_dict.keys())
        local_classes = set(client_train_classes[client_id])
        non_local_classes = all_classes - local_classes
        
        if not non_local_classes:  # Client has all classes locally
            kte_results[client_id] = 1.0
            continue
            
        # Calculate improvement on non-local classes
        non_local_initial_acc = np.mean([initial_accuracies[client_id][cls] for cls in non_local_classes])
        non_local_final_acc = np.mean([final_acc_dict[cls] for cls in non_local_classes])
        
        # Calculate improvement on local classes
        if local_classes:
            local_initial_acc = np.mean([initial_accuracies[client_id][cls] for cls in local_classes])
            local_final_acc = np.mean([final_acc_dict[cls] for cls in local_classes])
            local_improvement = max(local_final_acc - local_initial_acc, 0)
        else:
            local_improvement = 0
            
        non_local_improvement = max(non_local_final_acc - non_local_initial_acc, 0)
        
        # KTE is ratio of non-local to local improvement (normalized to [0,1])
        if local_improvement > 0:
            kte = non_local_improvement / local_improvement
            kte = min(max(kte, 0), 1)  # Clamp to [0,1]
        else:
            kte = 1.0 if non_local_improvement > 0 else 0.0
            
        kte_results[client_id] = kte
    
    return kte_results


def calculate_catastrophic_forgetting_metric(client_accuracies_history, client_train_classes, round_indices):
    """
    Catastrophic Forgetting Metric (CFM): Measures how much knowledge of existing classes 
    is lost when new clients are added.
    
    Args:
        client_accuracies_history: History of client accuracies per round (per class)
        client_train_classes: Dictionary of classes available to each client
        round_indices: Indices of rounds when new clients were added
        
    Returns:
        CFM values for each client transition
    """
    cfm_results = {}
    
    # For each client addition event
    for i in range(1, len(round_indices)):
        prev_round = round_indices[i-1]
        curr_round = round_indices[i]
        
        client_added = None
        for client_id in range(1, len(client_accuracies_history)+1):
            if client_id not in client_accuracies_history:
                continue
                
            # Find which client was added in this round
            prev_clients = set(client_train_classes.keys())
            curr_clients = set(client_train_classes.keys())
            new_clients = curr_clients - prev_clients
            
            if new_clients:
                client_added = list(new_clients)[0]
                break
        
        if client_added is None:
            continue
            
        # For each existing client, calculate performance drop on their trained classes
        forgetting_metrics = {}
        for client_id, acc_history in client_accuracies_history.items():
            if client_id == client_added:
                continue
                
            # Get classes that this client trained on
            local_classes = set(client_train_classes[client_id])
            
            # Calculate average accuracy before and after new client was added
            if prev_round < len(acc_history) and curr_round < len(acc_history):
                prev_acc = np.mean([acc_history[prev_round][cls] for cls in local_classes])
                curr_acc = np.mean([acc_history[curr_round][cls] for cls in local_classes])
                
                # Forgetting is measured as accuracy drop (clamped to positive values)
                forgetting = max(prev_acc - curr_acc, 0)
                
                # Normalize to [0,1] range (100% drop would be 1.0)
                forgetting_metrics[client_id] = min(forgetting / prev_acc if prev_acc > 0 else 0, 1.0)
            else:
                forgetting_metrics[client_id] = 0.0
                
        # Average forgetting across all existing clients
        if forgetting_metrics:
            cfm_results[client_added] = np.mean(list(forgetting_metrics.values()))
        else:
            cfm_results[client_added] = 0.0
    
    return cfm_results


def calculate_communication_efficiency(model_size_bytes, num_parameters, client_sequence_history):
    """
    Communication Efficiency (CE): Measures communication overhead in federated learning.
    
    Args:
        model_size_bytes: Size of the model in bytes
        num_parameters: Number of model parameters
        client_sequence_history: History of client participation
        
    Returns:
        Dictionary of communication efficiency metrics
    """
    # Calculate total bytes transferred in both directions
    total_rounds = len(client_sequence_history)
    total_client_rounds = sum(len(clients) for clients in client_sequence_history)
    
    # Each client downloads global model and uploads local model
    total_bytes_transferred = 2 * model_size_bytes * total_client_rounds
    
    # Calculate bytes per parameter per client
    bytes_per_param_per_client = total_bytes_transferred / (num_parameters * total_client_rounds)
    
    # Calculate communication efficiency (inverse of bytes transferred)
    efficiency = 1.0 / (bytes_per_param_per_client + 1e-6)  # Add epsilon to avoid division by zero
    
    return {
        'total_bytes_transferred': total_bytes_transferred,
        'bytes_per_parameter_per_client': bytes_per_param_per_client,
        'communication_efficiency': efficiency
    }


def sequential_federated_learning():
    # Configuration
    CONFIG = {
        'root_dir': 'ProcessedData50Refin',
        'batch_size': 16,
        'learning_rate': 0.0001,
        'sequence_length': 8,
        'num_workers': 4,
        'client_epochs': 10,
        'federated_rounds': 15,
        'image_size': 64,
        'num_clients': 8,
        'train_split': 0.8,
        'min_classes_per_client': 3,  # Minimum number of classes per client (except client 1)
        'sequential_threshold': 0.65  # Performance threshold to move to next client
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # Create save directory
    os.makedirs('sequential_federated_checkpoints', exist_ok=True)

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Define action class names
    action_class_names = [
        'Bend Down', 'Getting Up', 'Jumping', 'Left',
        'Move Back', 'Move Forward Fast', 'Move Forward Slow',
        'Right', 'Stop', 'Turn Around', 'Walking'
    ]

    # Initialize global model with attention mechanism
    global_model = EnhancedActionRecognitionModel(num_classes=len(action_class_names)).to(device)

    # Define class distribution for each client (for training)
    # Client 1 has all classes, others have subsets
    num_classes = len(action_class_names)
    np.random.seed(42)  # For reproducibility
    
    client_train_classes = {}
    client_train_classes[1] = list(range(num_classes))  # Client 1 has all classes
    
    # For other clients, assign a random subset of classes (minimum 3)
    for client_id in range(2, CONFIG['num_clients'] + 1):
        num_classes_for_client = np.random.randint(
            CONFIG['min_classes_per_client'],
            num_classes - 1  # At least one class different from client 1
        )
        client_train_classes[client_id] = np.random.choice(
            num_classes,
            size=num_classes_for_client,
            replace=False
        ).tolist()
    
    # Plot class distribution
    plot_class_distribution(client_train_classes, action_class_names, 'class_distribution.png')
    
    # Create client datasets with train/test split
    client_train_datasets = {}
    client_test_datasets = {}
    
    for client_id in range(1, CONFIG['num_clients'] + 1):
        client_path = os.path.join(CONFIG['root_dir'], f'Client {client_id}')
        
        # For test datasets, include all classes
        full_test_dataset = ClientDataset(
            client_dir=client_path,
            sequence_length=CONFIG['sequence_length'],
            transform=transform
        )
        
        # For training datasets, use only specified classes for each client
        full_train_dataset = ClientDataset(
            client_dir=client_path,
            sequence_length=CONFIG['sequence_length'],
            transform=transform,
            available_classes=client_train_classes[client_id]
        )
        
        # Create train/test splits
        train_size = int(CONFIG['train_split'] * len(full_train_dataset))
        test_size = len(full_test_dataset) - train_size
        
        # For training, use the dataset with filtered classes
        train_dataset = full_train_dataset
        
        # For testing, use all classes but create a subset for proper splits
        test_indices = list(range(len(full_test_dataset)))
        test_dataset = Subset(full_test_dataset, test_indices[-test_size:])
        
        client_train_datasets[client_id] = train_dataset
        client_test_datasets[client_id] = test_dataset
        
        logger.info(f"Client {client_id} - Train Classes: {client_train_classes[client_id]}")
        logger.info(f"Client {client_id} - Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    # Initialize tracking for test accuracies
    round_test_accuracies = []
    client_test_accuracies = {client_id: [] for client_id in range(1, CONFIG['num_clients'] + 1)}
    per_class_accuracies = {client_id: [] for client_id in range(1, CONFIG['num_clients'] + 1)}
    
    # For new metrics
    initial_per_class_accuracies = {}
    client_addition_rounds = []
    all_rounds_per_class_acc = {client_id: [] for client_id in range(1, CONFIG['num_clients'] + 1)}
    
    # Sequential federated learning
    criterion = nn.CrossEntropyLoss()
    current_client_sequence = []  # Keep track of clients processed
    
    # Start with client 1 (which has all classes)
    active_clients = [1]
    client_sequence_history = [active_clients.copy()]  # Record client participation history
    all_client_histories = {}
    
    # For Communication Efficiency metric
    model_size_bytes = sum(p.nelement() * p.element_size() for p in global_model.parameters())
    num_parameters = sum(p.nelement() for p in global_model.parameters())
    logger.info(f"Model size: {model_size_bytes / (1024 * 1024):.2f} MB, Parameters: {num_parameters:,}")
    
    for round in range(CONFIG['federated_rounds']):
        logger.info(f"\n{'='*50}")
        logger.info(f"Sequential Federated Learning Round {round + 1}/{CONFIG['federated_rounds']}")
        logger.info(f"Active clients: {active_clients}")
        logger.info(f"{'='*50}")

        client_models = []
        client_weights = []  # For weighted averaging
        round_histories = {}

        # Train on active clients
        for client_id in active_clients:
            logger.info(f"\nTraining on Client {client_id}")
            logger.info(f"Available classes: {client_train_classes[client_id]}")

            client_loader = DataLoader(
                client_train_datasets[client_id],
                batch_size=CONFIG['batch_size'],
                shuffle=True,
                num_workers=CONFIG['num_workers']
            )

            client_model = copy.deepcopy(global_model)
            optimizer = torch.optim.AdamW(
                client_model.parameters(),
                lr=CONFIG['learning_rate']
            )

            client_state_dict, history = train_client(
                model=client_model,
                train_loader=client_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                epochs=CONFIG['client_epochs']
            )

            # Weight by the number of classes this client has (more classes = higher weight)
            weight = len(client_train_classes[client_id]) / num_classes
            
            client_models.append(client_state_dict)
            client_weights.append(weight)
            round_histories[client_id] = history

        # Update global model with weighted averaging
        averaged_state_dict = weighted_average_models(client_models, client_weights)
        global_model.load_state_dict(averaged_state_dict)

        # Evaluate on each client's test set after this round
        logger.info(f"\nRound {round + 1} Test Performance:")
        logger.info("-" * 30)
        
        round_client_accuracies = []
        round_per_class_acc = {class_idx: [] for class_idx in range(num_classes)}

        for client_id in range(1, CONFIG['num_clients'] + 1):  # Evaluate on ALL clients
            test_loader = DataLoader(
                client_test_datasets[client_id],
                batch_size=CONFIG['batch_size'],
                shuffle=False,
                num_workers=CONFIG['num_workers']
            )
            
            test_loss, test_acc, pred_labels, true_labels, class_accuracies = evaluate_model(
                model=global_model,
                test_loader=test_loader,
                criterion=criterion,
                device=device,
                class_names=action_class_names
            )
            
            round_client_accuracies.append(test_acc)
            client_test_accuracies[client_id].append(test_acc)
            per_class_accuracies[client_id].append(class_accuracies)
            
            # Store initial accuracies for KTE metric if this is the first round
            if round == 0:
                initial_per_class_accuracies[client_id] = class_accuracies.copy()
                
            # Store per-class accuracies for all rounds for CFM metric
            all_rounds_per_class_acc[client_id].append(class_accuracies.copy())
            
            # Aggregate per-class accuracies
            for class_idx, acc in class_accuracies.items():
                round_per_class_acc[class_idx].append(acc)
            
            # Generate confusion matrix for this client
            if client_id in active_clients:
                plot_confusion_matrix(
                    true_labels, 
                    pred_labels, 
                    action_class_names,
                    f'sequential_federated_checkpoints/round_{round+1}_client_{client_id}_confusion.png'
                )
            
            active_status = "ACTIVE" if client_id in active_clients else "inactive"
            logger.info(f"Client {client_id:<2} ({active_status}) - Accuracy: {test_acc:.2f}%")
            
            # Log per-class performance for this client
            logger.info(f"  Per-class accuracies:")
            for class_idx, acc in class_accuracies.items():
                class_name = action_class_names[class_idx]
                in_training = class_idx in client_train_classes[client_id]
                logger.info(f"    {class_name:<15}: {acc:.2f}% {'(in training)' if in_training else ''}")

        # Calculate and store average test accuracy
        avg_round_accuracy = np.mean(round_client_accuracies)
        round_test_accuracies.append(avg_round_accuracy)
        
        # Calculate per-class average accuracies across all clients
        avg_per_class_acc = {class_idx: np.mean(accs) for class_idx, accs in round_per_class_acc.items()}
        
        logger.info("-" * 30)
        logger.info(f"Round {round + 1} Average Test Accuracy: {avg_round_accuracy:.2f}%")
        logger.info(f"Per-class average accuracies:")
        for class_idx, avg_acc in avg_per_class_acc.items():
            logger.info(f"  {action_class_names[class_idx]:<15}: {avg_acc:.2f}%")

        # Sequential client selection logic:
        # If average accuracy is above threshold, consider adding the next client
        if avg_round_accuracy > CONFIG['sequential_threshold'] * 100:
            # Find the next client that isn't already active
            next_clients = [c for c in range(1, CONFIG['num_clients'] + 1) if c not in active_clients]
            if next_clients:
                next_client = next_clients[0]
                active_clients.append(next_client)
                logger.info(f"Performance threshold reached! Adding Client {next_client} for the next round.")
                client_addition_rounds.append(round)  # For CFM metric
        
        client_sequence_history.append(active_clients.copy())

        # Save checkpoint
        checkpoint_path = os.path.join('sequential_federated_checkpoints', f'round_{round + 1}.pth')
        torch.save({
            'round': round + 1,
            'model_state_dict': global_model.state_dict(),
            'client_histories': round_histories,
            'active_clients': active_clients,
            'client_sequence_history': client_sequence_history,
            'test_accuracies': {
                'round_average': avg_round_accuracy,
                'client_accuracies': {client_id: acc for client_id, acc in zip(range(1, CONFIG['num_clients'] + 1), round_client_accuracies)},
                'per_class_accuracies': avg_per_class_acc
            }
        }, checkpoint_path)

        all_client_histories[round] = round_histories

    # Plot client participation over rounds
    plt.figure(figsize=(10, 6))
    rounds = list(range(1, CONFIG['federated_rounds'] + 2))  # +2 for initial and final states
    
    for client_id in range(1, CONFIG['num_clients'] + 1):
        client_participation = [1 if client_id in clients else 0 for clients in client_sequence_history]
        plt.plot(rounds, client_participation, 'o-', label=f'Client {client_id}')
    
    plt.xlabel('Round')
    plt.ylabel('Participation')
    plt.yticks([0, 1], ['Inactive', 'Active'])
    plt.title('Client Participation Over Rounds')
    plt.legend()
    plt.grid(True)
    plt.savefig('client_participation.png')
    plt.close()
    
    # Calculate specialized federated learning metrics
    logger.info("\n" + "="*50)
    logger.info("Specialized Federated Learning Metrics")
    logger.info("="*50)
    
    # 1. Client Convergence Rate (CCR)
    ccr_results = calculate_client_convergence_rate(all_client_histories[CONFIG['federated_rounds']-1])
    logger.info("\nClient Convergence Rate (lower is better):")
    for client_id, ccr in ccr_results.items():
        logger.info(f"Client {client_id}: {ccr:.4f}")
    
    # 2. Knowledge Transfer Efficiency (KTE)
    kte_results = calculate_knowledge_transfer_efficiency(
        initial_per_class_accuracies, 
        {client_id: per_class_accuracies[client_id][-1] for client_id in per_class_accuracies},
        client_train_classes
    )
    logger.info("\nKnowledge Transfer Efficiency (higher is better):")
    for client_id, kte in kte_results.items():
        logger.info(f"Client {client_id}: {kte:.4f}")
    
    # 3. Catastrophic Forgetting Metric (CFM)
    if client_addition_rounds:
        cfm_results = calculate_catastrophic_forgetting_metric(
            all_rounds_per_class_acc,
            client_train_classes,
            client_addition_rounds
        )
        logger.info("\nCatastrophic Forgetting Metric (lower is better):")
        for client_id, cfm in cfm_results.items():
            logger.info(f"When Client {client_id} was added: {cfm:.4f}")
    else:
        cfm_results = {}
    
    # 4. Communication Efficiency (CE)
    ce_results = calculate_communication_efficiency(
        model_size_bytes,
        num_parameters,
        client_sequence_history
    )
    logger.info("\nCommunication Efficiency:")
    logger.info(f"Total bytes transferred: {ce_results['total_bytes_transferred'] / (1024*1024):.2f} MB")
    logger.info(f"Bytes per parameter per client: {ce_results['bytes_per_parameter_per_client']:.4f}")
    logger.info(f"Communication efficiency score: {ce_results['communication_efficiency']:.6f}")
    
    # Plot metrics
    plt.figure(figsize=(15, 10))
    
    # Plot CCR
    plt.subplot(2, 2, 1)
    client_ids = list(ccr_results.keys())
    ccr_values = list(ccr_results.values())
    plt.bar(client_ids, ccr_values)
    plt.xlabel('Client ID')
    plt.ylabel('Convergence Rate')
    plt.title('Client Convergence Rate (lower is better)')
    
    # Plot KTE
    plt.subplot(2, 2, 2)
    client_ids = list(kte_results.keys())
    kte_values = list(kte_results.values())
    plt.bar(client_ids, kte_values)
    plt.xlabel('Client ID')
    plt.ylabel('Knowledge Transfer Efficiency')
    plt.title('Knowledge Transfer Efficiency (higher is better)')
    
    # Plot CFM if available
    if client_addition_rounds:
        plt.subplot(2, 2, 3)
        client_ids = list(cfm_results.keys())
        cfm_values = list(cfm_results.values())
        plt.bar(client_ids, cfm_values)
        plt.xlabel('Client ID Added')
        plt.ylabel('Forgetting Metric')
        plt.title('Catastrophic Forgetting Metric (lower is better)')
    
    plt.tight_layout()
    plt.savefig('federated_learning_metrics.png')
    plt.close()
    
    # Log final performance summary
    logger.info("\n" + "="*50)
    logger.info("Final Performance Summary")
    logger.info("="*50)
    logger.info(f"Final Average Test Accuracy: {round_test_accuracies[-1]:.2f}%")
    logger.info("\nClient Progress (First Round ? Final Round):")
    for client_id in client_test_accuracies:
        first_acc = client_test_accuracies[client_id][0]
        final_acc = client_test_accuracies[client_id][-1]
        improvement = final_acc - first_acc
        logger.info(f"Client {client_id:<2}: {first_acc:.2f}% ? {final_acc:.2f}% (Change: {improvement:+.2f}%)")
    
    # Save final model and results
    torch.save({
        'model_state_dict': global_model.state_dict(),
        'config': CONFIG,
        'client_histories': all_client_histories,
        'client_train_classes': client_train_classes,
        'client_sequence_history': client_sequence_history,
        'test_accuracies': {
            'round_accuracies': round_test_accuracies,
            'client_accuracies': client_test_accuracies,
            'per_class_accuracies': per_class_accuracies
        },
        'federated_metrics': {
            'ccr': ccr_results,
            'kte': kte_results,
            'cfm': cfm_results,
            'ce': ce_results
        }
    }, 'final_sequential_federated_model.pth')
    
    return global_model, all_client_histories, round_test_accuracies, client_test_accuracies, client_sequence_history, {
        'ccr': ccr_results,
        'kte': kte_results,
        'cfm': cfm_results,
        'ce': ce_results
    }


if __name__ == "__main__":
    try:
        model, histories, round_accuracies, client_accuracies, client_sequence, metrics = sequential_federated_learning()
        logger.info("\nSequential Federated Learning completed successfully!")
        
        # Print summary of specialized metrics
        logger.info("\n" + "="*50)
        logger.info("Specialized Federated Learning Metrics Summary")
        logger.info("="*50)
        
        # Average CCR across clients
        avg_ccr = np.mean(list(metrics['ccr'].values()))
        logger.info(f"Average Client Convergence Rate: {avg_ccr:.4f}")
        
        # Average KTE across clients
        avg_kte = np.mean(list(metrics['kte'].values()))
        logger.info(f"Average Knowledge Transfer Efficiency: {avg_kte:.4f}")
        
        # Average CFM across client additions (if any)
        if metrics['cfm']:
            avg_cfm = np.mean(list(metrics['cfm'].values()))
            logger.info(f"Average Catastrophic Forgetting Metric: {avg_cfm:.4f}")
        
        # Communication Efficiency summary
        logger.info(f"Total communication cost: {metrics['ce']['total_bytes_transferred'] / (1024*1024):.2f} MB")
        logger.info(f"Communication efficiency score: {metrics['ce']['communication_efficiency']:.6f}")

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user!")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()