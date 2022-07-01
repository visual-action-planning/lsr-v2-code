import torch
import importlib
import sys
sys.path.append('../architectures/')


def init_vae(opt):
    """Loads a pretrained VAE model."""
    path_to_pretrained = './models/{0}/vae_model.pt'.format(opt['exp_name'])
    vae_module = importlib.import_module("architectures.{0}".format(opt['model'])) # just have {0} if running the file directly
    print(' *- Imported module: ', vae_module)
    
    try:
        class_ = getattr(vae_module, opt['model'])
        vae_instance = class_(opt).to(opt['device'])
        print(' *- Loaded {0}.'.format(class_))
    except: 
        raise NotImplementedError(
                'Model {0} not recognized'.format(opt['model']))
    
    if opt['vae_load_checkpoint']:
        checkpoint = torch.load(path_to_pretrained, map_location=opt['device'])
        vae_instance.load_state_dict(checkpoint['model_state_dict'])
        print(' *- Loaded checkpoint.')
    else:
        vae_instance.load_state_dict(torch.load(path_to_pretrained, map_location=opt['device']))
    vae_instance.eval()
    assert(not vae_instance.training)
    return vae_instance

    
def vae_forward_pass(img, vae, opt):
    """Returns latent samples from a trained VAE."""
    enc_mean, enc_logvar = vae.encoder(img)
    enc_std = torch.exp(0.5*enc_logvar)
    latent_normal = torch.distributions.normal.Normal(enc_mean, enc_std)
    
    z_samples = latent_normal.sample((opt['n_latent_samples'], ))
    if opt['n_latent_samples'] > 1:
        z_samples = z_samples.squeeze()
    
    dec_mean_samples, _ = vae.decoder(z_samples)
    dec_mean_samples = dec_mean_samples.detach()
    
    dec_mean_original, _ = vae.decoder(enc_mean)
    dec_mean_original = dec_mean_original.detach()
    return enc_mean, z_samples, dec_mean_original, dec_mean_samples
    
