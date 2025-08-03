import torch
import time
from .utils import process_img

def test(env, model, device, num_episodes=100):
    model = model.to(device)
    model.eval()

    for episode in range(num_episodes):
        obs, info = env.reset()

        img_state = process_img(obs)
        state = torch.cat((img_state, img_state, img_state, img_state)).unsqueeze(0).to(device)

        while True:

            with torch.no_grad():
                q_values = model(state)
                action = q_values.argmax(dim=1).item()

            next_obs, reward, terminated, truncated, info = env.step(action)

            img_next_state = process_img(next_obs).to(device)
            next_state = torch.cat((state.squeeze(0)[1:, :, :], img_next_state)).unsqueeze(0).to(device)
            
            env.render()
            time.sleep(0.05)

            if terminated or truncated:

                if info['score']:
                    print(f'Episode {episode + 1}, score {info["score"]}')
                break