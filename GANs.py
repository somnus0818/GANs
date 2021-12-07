import torch
import torchvision
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np

# InlineBackend.figure_format = "retina"

epoch_n = 20
batchsize = 128
smooth = 0.1

train_transform = transforms.ToTensor()

train_data = datasets.MNIST(root="data", download=True, train=True, transform=train_transform)
train_load = torch.utils.data.DataLoader(dataset=train_data, shuffle=True, batch_size=batchsize)


def plot_img(img):
    img = torchvision.utils.make_grid(img)
    img = img.numpy().transpose(1, 2, 0)
    plt.figure(figsize=(12, 9))
    plt.imshow(img)


class Discriminator_conv(torch.nn.Module):

    def __init__(self):
        super(Discriminator_conv, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=5, stride=1),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(32, 64, kernel_size=5, stride=1),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(64 * 4 * 4, 64 * 4 * 4),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64 * 4 * 4, 1)
        )

    def forward(self, input):
        output = self.conv(input)
        output = output.view(-1, 64 * 4 * 4)
        output = self.dense(output)
        return output


class Generator_conv(torch.nn.Module):

    def __init__(self):
        super(Generator_conv, self).__init__()
        self.conv_dense = torch.nn.Sequential(
            torch.nn.Linear(100, 1024),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm1d(num_features=1024),
            torch.nn.Linear(1024, 7 * 7 * 128),
            torch.nn.BatchNorm1d(num_features=7 * 7 * 128)
        )
        self.transpose_conv = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            torch.nn.Tanh()
        )

    def forward(self, input):
        output = self.conv_dense(input)
        output = output.view(-1, 128, 7, 7)
        output = self.transpose_conv(output)
        return output


def initialize_weights(m):
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight.data)


model_discriminator = Discriminator_conv().cuda()
model_discriminator.apply(initialize_weights)
model_generator = Generator_conv().cuda()
model_generator.apply(initialize_weights)

loss_f = torch.nn.BCEWithLogitsLoss()

optimizer_dis = torch.optim.Adam(model_discriminator.parameters(), lr=0.0001)
optimizer_gen = torch.optim.Adam(model_generator.parameters(), lr=0.0001)

samples = []
losses = []


def rand_img(batchsize, output_size):
    Z = np.random.uniform(-1., 1., size=(batchsize, output_size))
    Z = np.float32(Z)
    Z = torch.from_numpy(Z)
    Z = Variable(Z.cuda())
    return Z


for epoch in range(epoch_n):

    for batch in train_load:
        X_train, y_train = batch
        X_train, y_train = Variable(X_train.cuda()), Variable(y_train.cuda())
        # X_train,y_train = Variable(X_train),Variable(y_train)
        Z = rand_img(batchsize=batchsize, output_size=100)

        optimizer_dis.zero_grad()
        X_gen = model_generator(Z)
        X_gen = X_gen.view(-1, 1, 28, 28)
        X_train = X_train.view(-1, 1, 28, 28)

        logits_real = model_discriminator(X_train)
        logits_fake = model_discriminator(X_gen)

        d_loss = loss_f(logits_real, torch.ones_like(logits_real) * (1 - smooth)) + loss_f(logits_fake,
                                                                                           torch.zeros_like(
                                                                                               logits_fake))
        d_loss.backward(retain_graph=True)
        optimizer_dis.step()

        optimizer_gen.zero_grad()

        Z = rand_img(batchsize=batchsize, output_size=100)
        X_gen = model_generator(Z)
        X_gen = X_gen.view(-1, 1, 28, 28)
        logits_fake = model_discriminator(X_gen)
        g_loss = loss_f(logits_fake, torch.ones_like(logits_fake))
        g_loss.backward()
        optimizer_gen.step()

    print("Epoch{}/{}...".format(epoch + 1, epoch_n),
          "Discriminator Loss:{:.4f}...".format(d_loss),
          "Generator Loss:{:.4f}...".format(g_loss))

    losses.append((d_loss, g_loss))

    fake_img = model_generator(Z)
    samples.append(fake_img)

fig, ax = plt.subplots()
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator')
plt.plot(losses.T[1], label='Generator')
plt.title("Training Losses")
plt.legend()


def to_img(img):
    img = img.detach().cpu().data
    img = img.clamp(0, 1)
    img = img.view(-1, 1, 28, 28)
    return img


for i in range(len(samples)):
    img = to_img(samples[i])
    plot_img(img)