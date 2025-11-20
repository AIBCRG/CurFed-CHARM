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
from scipy.stats import entropy, wasserstein_distance
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
            'Bend Down': 0,
            'Getting Up': 1,
            'Jumping': 2,
            'Left': 3,
            'Move Back': 4,
            'Move Forward Fast': 5,
            'Move Forward Slow': 6,
            'Right': 7,
            'Stop': 8,
            'Turn Around': 9,
            'Walking': 10
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
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        # Add attention mechanism
        self.attention = AttentionBlock(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
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


def train_client(model, train_loader, criterion, optimizer, device, epochs=5, 
                lambda_weights=None, mu_r=0.0, w_prev=None):
    """
    Train client with weighted loss and regularization
    
    Args:
        lambda_weights: dict with 'lambda_1', 'lambda_2', 'lambda_3'
        mu_r: adaptive regularization coefficient for proximal term
        w_prev: previous global model parameters
    """
    model.train()
    history = {'loss': [], 'accuracy': []}
    
    if lambda_weights is None:
        lambda_weights = {'lambda_1': 1.0, 'lambda_2': 0.0, 'lambda_3': 0.0}
    
    for epoch in range(epochs):
        running_loss = 0.0
        running_ce_loss = 0.0
        running_l2_loss = 0.0
        running_aux_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # Loss Component 1: Cross-entropy loss (main classification loss)
            loss_ce = criterion(outputs, labels)

            # Loss Component 2: L2 Regularization (prevents overfitting)
            l2_reg = torch.tensor(0., device=device)
            for param in model.parameters():
                l2_reg += torch.norm(param, p=2)
            loss_l2 = l2_reg / max(len(list(model.parameters())), 1)

            # Loss Component 3: Proximal term (prevents client drift)
            loss_prox = torch.tensor(0., device=device)
            if w_prev is not None and mu_r > 0:
                for (name, param), (prev_name, prev_param) in zip(
                    model.named_parameters(), w_prev.named_parameters()
                ):
                    loss_prox += torch.norm(param - prev_param, p=2)
                loss_prox = loss_prox / max(len(list(model.parameters())), 1)

            # Weighted combination of losses
            lambda_1 = lambda_weights.get('lambda_1', 1.0)
            lambda_2 = lambda_weights.get('lambda_2', 0.0)
            lambda_3 = lambda_weights.get('lambda_3', 0.0)

            total_loss = (lambda_1 * loss_ce + 
                         lambda_2 * loss_l2 + 
                         lambda_3 * loss_prox)

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            running_ce_loss += loss_ce.item()
            running_l2_loss += loss_l2.item()
            running_aux_loss += loss_prox.item()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_ce_loss = running_ce_loss / len(train_loader)
        epoch_l2_loss = running_l2_loss / len(train_loader)
        epoch_aux_loss = running_aux_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)

        logger.info(f'Epoch {epoch + 1}/{epochs}: '
                   f'Total Loss={epoch_loss:.4f}, '
                   f'CE Loss={epoch_ce_loss:.4f}, '
                   f'L2 Loss={epoch_l2_loss:.4f}, '
                   f'Prox Loss={epoch_aux_loss:.4f}, '
                   f'Acc={epoch_acc:.2f}%')

    return model.state_dict(), history


def calculate_client_difficulty_score(client_id, active_clients, all_classes, 
                                     client_train_classes, lambda_1=0.5, 
                                     lambda_2=0.3, lambda_3=0.2):
    # Component 1: Class Uniqueness (?1)
    # |C? ? ???S? C?| / |C?|
    client_classes = set(client_train_classes[client_id])
    active_classes = set()
    for active_client in active_clients:
        active_classes.update(client_train_classes[active_client])
    
    unique_classes = client_classes - active_classes
    class_uniqueness = len(unique_classes) / max(len(client_classes), 1)
    
    # Component 2: Distribution Entropy (?2)
    # H(p?) - Shannon entropy of class distribution
    class_counts = np.ones(len(all_classes))  # Assume uniform distribution
    # Normalize to probabilities
    class_dist = class_counts / np.sum(class_counts)
    distribution_entropy = entropy(class_dist)
    
    # Normalize entropy to [0, 1]
    max_entropy = np.log(len(all_classes))
    normalized_entropy = distribution_entropy / max(max_entropy, 1e-6)
    
    # Component 3: Domain Gap (?3)
    # D(p?, p??) - Wasserstein distance between distributions
    client_dist = np.zeros(len(all_classes))
    for cls in client_classes:
        client_dist[cls] = 1.0 / len(client_classes)
    
    active_dist = np.zeros(len(all_classes))
    for cls in active_classes:
        active_dist[cls] = 1.0 / max(len(active_classes), 1)
    
    # Calculate Wasserstein distance
    domain_gap = wasserstein_distance(client_dist, active_dist)
    
    # Normalize domain gap to [0, 1]
    normalized_domain_gap = min(domain_gap, 1.0)
    
    # Calculate final difficulty score
    difficulty_score = (lambda_1 * class_uniqueness +
                       lambda_2 * normalized_entropy +
                       lambda_3 * normalized_domain_gap)
    
    logger.info(f"Client {client_id} Difficulty Score Components:")
    logger.info(f"  ?1 (Class Uniqueness): {lambda_1}  {class_uniqueness:.4f} = {lambda_1 * class_uniqueness:.4f}")
    logger.info(f"  ?2 (Distribution Entropy): {lambda_2}  {normalized_entropy:.4f} = {lambda_2 * normalized_entropy:.4f}")
    logger.info(f"  ?3 (Domain Gap): {lambda_3}  {normalized_domain_gap:.4f} = {lambda_3 * normalized_domain_gap:.4f}")
    logger.info(f"  Total Difficulty Score: {difficulty_score:.4f}")
    
    return difficulty_score


def calculate_curriculum_weight(client_id, active_clients, current_round, 
                               client_addition_rounds, adaptation_period=3):
    """
    Calculate curriculum-specific coefficient ??? per Equation 14:
    ??? = {
        1.0,                         if i has been in S? for more than ? rounds
        exp((r? - r + ?) / ?),      if r - r? < ?
    }
    
    Args:
        client_id: client ID
        active_clients: currently active clients
        current_round: current training round
        client_addition_rounds: dict mapping client_id to round when added
        adaptation_period: ? parameter (rounds for adaptation)
    
    Returns:
        curriculum_weight: weight coefficient
    """
    if client_id not in client_addition_rounds:
        # Client has been there from start
        return 1.0
    
    round_added = client_addition_rounds[client_id]
    rounds_in_federation = current_round - round_added
    
    if rounds_in_federation >= adaptation_period:
        # Client has completed warm-up phase
        gamma = 1.0
    else:
        # Warm-up phase: gradual increase in influence
        gamma = np.exp((round_added - current_round + adaptation_period) / adaptation_period)
    
    return gamma


def weighted_average_models(models_dict_list, client_weights):
    """Average model parameters weighted by client contributions."""
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

            if class_names:
                for i, label in enumerate(labels):
                    class_idx = label.item()
                    class_total[class_idx] += 1
                    if predicted[i] == label:
                        class_correct[class_idx] += 1

    test_loss = running_loss / len(test_loader)
    test_acc = 100. * correct / total

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
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_training_curves(client_histories, save_path):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    for client_id, history in client_histories.items():
        plt.plot(history['loss'], label=f'Client {client_id}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss per Client')

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
    availability_matrix = np.zeros((len(client_classes), len(class_names)))
    for i, (client_id, available_classes) in enumerate(client_classes.items()):
        for class_idx in available_classes:
            availability_matrix[i, class_idx] = 1

    sns.heatmap(availability_matrix, annot=True, fmt='.0f', cmap='YlGnBu', 
                xticklabels=class_names, yticklabels=[f'Client {cid}' for cid in client_classes.keys()])
    plt.title('Class Distribution Across Clients')
    plt.xlabel('Action Classes')
    plt.ylabel('Clients')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def calculate_client_convergence_rate(client_histories):
    """Client Convergence Rate (CCR): Measures how quickly clients reach stable performance."""
    ccr_results = {}
    for client_id, history in client_histories.items():
        losses = history['loss']
        accuracies = history['accuracy']
        convergence_epoch = None
        for i in range(3, len(losses)):
            loss_change = abs(losses[i] - losses[i-3]) / max(losses[i-3], 1e-6)
            acc_change = abs(accuracies[i] - accuracies[i-3]) / max(accuracies[i-3], 1e-6)
            if loss_change < 0.01 and acc_change < 0.01:
                convergence_epoch = i
                break

        if convergence_epoch is not None:
            ccr = convergence_epoch / len(losses)
        else:
            ccr = 1.0

        ccr_results[client_id] = ccr
    return ccr_results


def calculate_knowledge_transfer_efficiency(initial_accuracies, final_accuracies, client_train_classes):
    """Knowledge Transfer Efficiency (KTE): Measures how well knowledge transfers to unseen classes."""
    kte_results = {}
    for client_id, final_acc_dict in final_accuracies.items():
        all_classes = set(final_acc_dict.keys())
        local_classes = set(client_train_classes[client_id])
        non_local_classes = all_classes - local_classes

        if not non_local_classes:
            kte_results[client_id] = 1.0
            continue

        non_local_initial_acc = np.mean([initial_accuracies[client_id][cls] for cls in non_local_classes])
        non_local_final_acc = np.mean([final_acc_dict[cls] for cls in non_local_classes])

        if local_classes:
            local_initial_acc = np.mean([initial_accuracies[client_id][cls] for cls in local_classes])
            local_final_acc = np.mean([final_acc_dict[cls] for cls in local_classes])
            local_improvement = max(local_final_acc - local_initial_acc, 0)
        else:
            local_improvement = 0

        non_local_improvement = max(non_local_final_acc - non_local_initial_acc, 0)

        if local_improvement > 0:
            kte = non_local_improvement / local_improvement
            kte = min(max(kte, 0), 1)
        else:
            kte = 1.0 if non_local_improvement > 0 else 0.0

        kte_results[client_id] = kte
    return kte_results


def calculate_catastrophic_forgetting_metric(client_accuracies_history, client_train_classes, round_indices):
    """Catastrophic Forgetting Metric (CFM): Measures knowledge loss when new clients are added."""
    cfm_results = {}
    
    for i in range(1, len(round_indices)):
        prev_round = round_indices[i-1]
        curr_round = round_indices[i]
        forgetting_metrics = {}
        
        for client_id, acc_history in client_accuracies_history.items():
            local_classes = set(client_train_classes[client_id])
            
            if prev_round < len(acc_history) and curr_round < len(acc_history):
                prev_acc = np.mean([acc_history[prev_round][cls] for cls in local_classes])
                curr_acc = np.mean([acc_history[curr_round][cls] for cls in local_classes])
                forgetting = max(prev_acc - curr_acc, 0)
                forgetting_metrics[client_id] = min(forgetting / prev_acc if prev_acc > 0 else 0, 1.0)
            else:
                forgetting_metrics[client_id] = 0.0

        if forgetting_metrics:
            cfm_results[i] = np.mean(list(forgetting_metrics.values()))
        else:
            cfm_results[i] = 0.0

    return cfm_results


def calculate_communication_efficiency(model_size_bytes, num_parameters, client_sequence_history):
    """Communication Efficiency (CE): Measures communication overhead in federated learning."""
    total_rounds = len(client_sequence_history)
    total_client_rounds = sum(len(clients) for clients in client_sequence_history)
    total_bytes_transferred = 2 * model_size_bytes * total_client_rounds
    bytes_per_param_per_client = total_bytes_transferred / (num_parameters * total_client_rounds)
    efficiency = 1.0 / (bytes_per_param_per_client + 1e-6)

    return {
        'total_bytes_transferred': total_bytes_transferred,
        'bytes_per_parameter_per_client': bytes_per_param_per_client,
        'communication_efficiency': efficiency
    }


def sequential_federated_learning():
    # ========================================================================
    # CONFIGURATION WITH ALL HYPERPARAMETERS
    # ========================================================================
    CONFIG = {
        'root_dir': 'ProcessedData50Refin',
        'batch_size': 8,
        'learning_rate': 0.0001,
        'sequence_length': 8,
        'num_workers': 4,
        'client_epochs': 5,
        'federated_rounds': 10,
        'image_size': 64,
        'num_clients': 8,
        'train_split': 0.8,
        'min_classes_per_client': 3,
        
        # ===== CURRICULUM LEARNING HYPERPARAMETERS =====
        # Performance threshold (t) - Accuracy threshold for adding clients
        'sequential_threshold': 0.75,
        
        # Client adaptation period (?) - Rounds for gradual client integration
        'client_adaptation_period': 3,
        
        # Weighting factors for difficulty score (?1, ?2, ?3)
        'lambda_1': 0.5,  # Weight for class uniqueness
        'lambda_2': 0.3,  # Weight for distribution entropy
        'lambda_3': 0.2,  # Weight for domain gap
        
        # Proximal term coefficients (FedProx)
        'mu_base': 0.01,      # Baseline regularization strength
        'mu_0': 0.1,          # Initial regularization for new clients
        'kappa': 3,           # Proximal term decay rate
    }

    logger.info("="*70)
    logger.info("CURRICULUM FEDERATED LEARNING CONFIGURATION")
    logger.info("="*70)
    logger.info(f"t (Performance Threshold): {CONFIG['sequential_threshold']}")
    logger.info(f"? (Adaptation Period): {CONFIG['client_adaptation_period']} rounds")
    logger.info(f"?1 (Class Uniqueness Weight): {CONFIG['lambda_1']}")
    logger.info(f"?2 (Distribution Entropy Weight): {CONFIG['lambda_2']}")
    logger.info(f"?3 (Domain Gap Weight): {CONFIG['lambda_3']}")
    logger.info("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # Create save directory
    os.makedirs('sequential_federated_checkpoints', exist_ok=True)

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Define action class names
    action_class_names = [
        'Bend Down', 'Getting Up', 'Jumping', 'Left', 'Move Back',
        'Move Forward Fast', 'Move Forward Slow', 'Right', 'Stop', 'Turn Around', 'Walking'
    ]

    # Initialize global model
    global_model = EnhancedActionRecognitionModel(num_classes=len(action_class_names)).to(device)

    # Define class distribution for each client
    num_classes = len(action_class_names)
    np.random.seed(42)
    client_train_classes = {}
    client_train_classes[1] = list(range(num_classes))
    for client_id in range(2, CONFIG['num_clients'] + 1):
        num_classes_for_client = np.random.randint(
            CONFIG['min_classes_per_client'],
            num_classes - 1
        )
        client_train_classes[client_id] = np.random.choice(
            num_classes, size=num_classes_for_client, replace=False
        ).tolist()

    # Plot class distribution
    plot_class_distribution(client_train_classes, action_class_names, 'class_distribution.png')

    # Create client datasets
    client_train_datasets = {}
    client_test_datasets = {}
    for client_id in range(1, CONFIG['num_clients'] + 1):
        client_path = os.path.join(CONFIG['root_dir'], f'Client {client_id}')
        full_test_dataset = ClientDataset(
            client_dir=client_path,
            sequence_length=CONFIG['sequence_length'],
            transform=transform
        )
        full_train_dataset = ClientDataset(
            client_dir=client_path,
            sequence_length=CONFIG['sequence_length'],
            transform=transform,
            available_classes=client_train_classes[client_id]
        )

        train_size = int(CONFIG['train_split'] * len(full_train_dataset))
        test_size = len(full_test_dataset) - train_size
        test_indices = list(range(len(full_test_dataset)))
        test_dataset = Subset(full_test_dataset, test_indices[-test_size:])

        client_train_datasets[client_id] = full_train_dataset
        client_test_datasets[client_id] = test_dataset

        logger.info(f"Client {client_id} - Train Classes: {client_train_classes[client_id]}")
        logger.info(f"Client {client_id} - Train: {len(full_train_dataset)}, Test: {len(test_dataset)}")

    # Initialize tracking
    round_test_accuracies = []
    client_test_accuracies = {client_id: [] for client_id in range(1, CONFIG['num_clients'] + 1)}
    per_class_accuracies = {client_id: [] for client_id in range(1, CONFIG['num_clients'] + 1)}
    initial_per_class_accuracies = {}
    client_addition_rounds = {}
    all_rounds_per_class_acc = {client_id: [] for client_id in range(1, CONFIG['num_clients'] + 1)}

    # Sequential federated learning
    criterion = nn.CrossEntropyLoss()
    active_clients = [1]
    client_sequence_history = [active_clients.copy()]
    all_client_histories = {}
    client_round_added = {}  # Track when each client was added

    # Model information
    model_size_bytes = sum(p.nelement() * p.element_size() for p in global_model.parameters())
    num_parameters = sum(p.nelement() for p in global_model.parameters())
    logger.info(f"Model size: {model_size_bytes / (1024 * 1024):.2f} MB, Parameters: {num_parameters:,}")

    # ========================================================================
    # MAIN FEDERATED LEARNING LOOP
    # ========================================================================
    for round_num in range(CONFIG['federated_rounds']):
        logger.info(f"\n{'='*70}")
        logger.info(f"Federated Learning Round {round_num + 1}/{CONFIG['federated_rounds']}")
        logger.info(f"Active clients: {active_clients}")
        logger.info(f"{'='*70}")

        client_models = []
        client_weights = []
        round_histories = {}

        # ====================================================================
        # TRAIN ON ACTIVE CLIENTS
        # ====================================================================
        for client_id in active_clients:
            logger.info(f"\nTraining Client {client_id}")
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

            if client_id in client_round_added:
                r_star = client_round_added[client_id]
                if round_num - r_star < CONFIG['kappa']:
                    mu_r = CONFIG['mu_0'] * np.exp(-(round_num - r_star) / CONFIG['kappa'])
                else:
                    mu_r = CONFIG['mu_base']
            else:
                mu_r = CONFIG['mu_base']

            # Calculate lambda weights for this client
            lambda_weights = {
                'lambda_1': CONFIG['lambda_1'],
                'lambda_2': CONFIG['lambda_2'],
                'lambda_3': CONFIG['lambda_3']
            }

            # Train client
            client_state_dict, history = train_client(
                model=client_model,
                train_loader=client_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                epochs=CONFIG['client_epochs'],
                lambda_weights=lambda_weights,
                mu_r=mu_r,
                w_prev=copy.deepcopy(global_model) if mu_r > 0 else None
            )

            # Weight by class diversity
            weight = len(client_train_classes[client_id]) / num_classes
            client_models.append(client_state_dict)
            client_weights.append(weight)
            round_histories[client_id] = history

        # ====================================================================
        # AGGREGATE GLOBAL MODEL
        # ====================================================================
        averaged_state_dict = weighted_average_models(client_models, client_weights)
        global_model.load_state_dict(averaged_state_dict)

        # ====================================================================
        # EVALUATE ON ALL CLIENTS
        # ====================================================================
        logger.info(f"\nRound {round_num + 1} Test Performance:")
        logger.info("-" * 70)

        round_client_accuracies = []
        round_per_class_acc = {class_idx: [] for class_idx in range(num_classes)}

        for client_id in range(1, CONFIG['num_clients'] + 1):
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

            if round_num == 0:
                initial_per_class_accuracies[client_id] = class_accuracies.copy()

            all_rounds_per_class_acc[client_id].append(class_accuracies.copy())

            for class_idx, acc in class_accuracies.items():
                round_per_class_acc[class_idx].append(acc)

            if client_id in active_clients:
                plot_confusion_matrix(
                    true_labels, pred_labels, action_class_names,
                    f'sequential_federated_checkpoints/round_{round_num+1}_client_{client_id}_confusion.png'
                )

            active_status = "ACTIVE" if client_id in active_clients else "inactive"
            logger.info(f"Client {client_id:<2} ({active_status}) - Accuracy: {test_acc:.2f}%")

        # Calculate average accuracy
        avg_round_accuracy = np.mean(round_client_accuracies)
        round_test_accuracies.append(avg_round_accuracy)

        logger.info("-" * 70)
        logger.info(f"Round {round_num + 1} Average Test Accuracy: {avg_round_accuracy:.2f}%")

        # ====================================================================
        # SEQUENTIAL CLIENT SELECTION WITH CURRICULUM
        # ====================================================================
        if avg_round_accuracy > CONFIG['sequential_threshold'] * 100:
            # Find next inactive client
            next_clients = [c for c in range(1, CONFIG['num_clients'] + 1) 
                           if c not in active_clients]
            
            if next_clients:
                # Calculate difficulty scores for candidate clients
                logger.info("\nCalculating difficulty scores for candidate clients...")
                difficulty_scores = {}
                
                for candidate_id in next_clients:
                    difficulty_score = calculate_client_difficulty_score(
                        client_id=candidate_id,
                        active_clients=active_clients,
                        all_classes=set(range(num_classes)),
                        client_train_classes=client_train_classes,
                        lambda_1=CONFIG['lambda_1'],
                        lambda_2=CONFIG['lambda_2'],
                        lambda_3=CONFIG['lambda_3']
                    )
                    difficulty_scores[candidate_id] = difficulty_score
                
                # Select client with lowest difficulty (easiest to learn)
                selected_client = min(difficulty_scores, key=difficulty_scores.get)
                
                active_clients.append(selected_client)
                client_round_added[selected_client] = round_num
                
                logger.info(f"\n{'='*70}")
                logger.info(f"Performance threshold t={CONFIG['sequential_threshold']} reached!")
                logger.info(f"Adding Client {selected_client} to federation")
                logger.info(f"Difficulty Score: {difficulty_scores[selected_client]:.4f}")
                logger.info(f"{'='*70}\n")
                
                client_addition_rounds[round_num] = selected_client
                client_sequence_history.append(active_clients.copy())

        # Save checkpoint
        checkpoint_path = os.path.join('sequential_federated_checkpoints', f'round_{round_num + 1}.pth')
        torch.save({
            'round': round_num + 1,
            'model_state_dict': global_model.state_dict(),
            'client_histories': round_histories,
            'active_clients': active_clients,
            'client_sequence_history': client_sequence_history,
            'test_accuracies': {
                'round_average': avg_round_accuracy,
                'client_accuracies': {client_id: acc for client_id, acc in zip(range(1, CONFIG['num_clients'] + 1), round_client_accuracies)},
            }
        }, checkpoint_path)

        all_client_histories[round_num] = round_histories

    # ========================================================================
    # PLOT CLIENT PARTICIPATION OVER ROUNDS
    # ========================================================================
    plt.figure(figsize=(10, 6))
    rounds = list(range(1, CONFIG['federated_rounds'] + 2))
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

    # ========================================================================
    # CALCULATE SPECIALIZED FEDERATED LEARNING METRICS
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("SPECIALIZED FEDERATED LEARNING METRICS")
    logger.info("="*70)

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
            list(client_addition_rounds.keys())
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

    # ========================================================================
    # PLOT METRICS
    # ========================================================================
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
    if cfm_results:
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

    # ========================================================================
    # FINAL PERFORMANCE SUMMARY
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("FINAL PERFORMANCE SUMMARY")
    logger.info("="*70)
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

    logger.info("\n" + "="*70)
    logger.info("Model and results saved to final_sequential_federated_model.pth")
    logger.info("="*70)

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
        logger.info("\n" + "="*70)
        logger.info("SPECIALIZED FEDERATED LEARNING METRICS SUMMARY")
        logger.info("="*70)

        avg_ccr = np.mean(list(metrics['ccr'].values()))
        logger.info(f"Average Client Convergence Rate: {avg_ccr:.4f}")

        avg_kte = np.mean(list(metrics['kte'].values()))
        logger.info(f"Average Knowledge Transfer Efficiency: {avg_kte:.4f}")

        if metrics['cfm']:
            avg_cfm = np.mean(list(metrics['cfm'].values()))
            logger.info(f"Average Catastrophic Forgetting Metric: {avg_cfm:.4f}")

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
