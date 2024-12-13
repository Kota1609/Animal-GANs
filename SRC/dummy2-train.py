import os
import torch
from torch import autograd
from torch.utils.data import DataLoader
from utils import create_custom_dataloader
from opt import parse_opt
from model import Generator, Discriminator
from torch.nn import DataParallel

def compute_gradient_penalty(discriminator, real_samples, fake_samples, Stru, Time, Dose, device):
    alpha = torch.rand((real_samples.size(0), 1), device=device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = discriminator(interpolates, Stru, Time, Dose)
    fake = torch.ones(real_samples.shape[0], 1).to(device)
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train(generator, discriminator, dataloader, n_epochs, n_critic, Z_dim, device, lr, b1, b2, interval, model_path,
          lambda_gp, lambda_GR):
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2), weight_decay=1e-4)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2), weight_decay=1e-4)

    for epoch in range(n_epochs):
        for i, (Measurement, Stru, Time, Dose) in enumerate(dataloader):
            batch_size = Measurement.shape[0]
            optimizer_D.zero_grad()

            # Generate noise and fake data
            z = torch.randn(batch_size, Z_dim, device=device)
            gen_Measurement = generator(z, Stru, Time, Dose)

            # Compute discriminator losses
            validity_real = discriminator(Measurement, Stru, Time, Dose)
            validity_fake = discriminator(gen_Measurement.detach(), Stru, Time, Dose)
            gradient_penalty = compute_gradient_penalty(discriminator, Measurement, gen_Measurement, Stru, Time, Dose,
                                                        device)
            d_loss = -torch.mean(validity_real) + torch.mean(validity_fake) + lambda_gp * gradient_penalty
            d_loss.backward()
            optimizer_D.step()

            # Train generator every n_critic steps
            if i % n_critic == 0:
                optimizer_G.zero_grad()
                gen_Measurement = generator(z, Stru, Time, Dose)
                validity = discriminator(gen_Measurement, Stru, Time, Dose)
                g_loss = -torch.mean(validity)
                g_loss.backward()
                optimizer_G.step()

            if i % 100 == 0:
                print(
                    f"[Epoch {epoch + 1}/{n_epochs}] [Batch {i + 1}/{len(dataloader)}] [D loss: {d_loss.item():.6f}] [G loss: {g_loss.item():.6f}]"
                )

        if (epoch + 1) % interval == 0:
            os.makedirs(model_path, exist_ok=True)
            torch.save(generator.state_dict(), os.path.join(model_path, f'generator_{epoch + 1}.pth'))


if __name__ == '__main__':
    opt = parse_opt()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataloader = create_custom_dataloader(opt.data_path, opt.descriptors_path, opt.batch_size, device, drop_last=True)

    generator = Generator(opt.Z_dim, opt.Stru_dim, opt.Time_dim, opt.Dose_dim, opt.Measurement_dim).to(device)
    discriminator = Discriminator(opt.Stru_dim, opt.Time_dim, opt.Dose_dim, opt.Measurement_dim).to(device)

    generator = DataParallel(generator)
    discriminator = DataParallel(discriminator)

    train(generator, discriminator, dataloader, opt.n_epochs, opt.n_critic, opt.Z_dim, device, opt.lr, opt.b1, opt.b2,
          opt.interval, opt.model_path, opt.lambda_gp, opt.lambda_GR)
