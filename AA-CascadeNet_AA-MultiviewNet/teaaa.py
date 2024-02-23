import torch 
import torch.nn.functional as F
torch.set_printoptions(profile="full")
labels = torch.cat([torch.arange(4) for _ in range(2)], dim=0)
print(labels)
print(labels.unsqueeze(0))
print(labels.unsqueeze(1))
labels = (labels.unsqueeze(0) == labels.unsqueeze(1))
print(labels)

features = torch.rand(4*2, 2)
features = F.normalize(features, dim=1)

print(features)

sim_mat = torch.matmul(features, features.T)
print(sim_mat)

assert sim_mat.shape == labels.shape, ('Labels and similarity matrix doesn`t have matching shapes.')
mask = torch.eye(labels.shape[0], dtype=torch.bool)

print(labels.shape)
labels = labels[~mask].view(labels.shape[0], -1)
print(labels.shape)
print(labels)
sim_mat = sim_mat[~mask].view(sim_mat.shape[0], -1)
print(sim_mat)

positives = sim_mat[labels.bool()].view(labels.shape[0], -1)
positive_labels = torch.ones(positives.shape[0], positives.shape[1])
print(positives.shape)
print(positive_labels)
negatives = sim_mat[~labels.bool()].view(sim_mat.shape[0], -1)
negative_labels = torch.zeros(negatives.shape[0], negatives.shape[1])
print(negatives.shape)
print(negative_labels)


logits = torch.cat([positives, negatives], dim=1)
# labels = torch.cat([positive_labels, negative_labels], dim=1)
print(logits)
print(labels)
labels = torch.zeros(logits.shape[0], dtype=torch.long)

print(labels.shape)

logits = logits / 0.7
print(logits.shape)

torch.exp(logits)
print(logits)

print(logits[:,0])
print(logits[~labels])
print(torch.sum(logits[:,1:8], dim=1))

res = -1 * torch.log(logits[:,0] / torch.sum(logits[:,1:8], dim=1))
print(res)

res2 = torch.sum(res)
print(res2)
