from collections import namedtuple
import random
import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


Transition = namedtuple('Transition',
                        ('attention_map', 'action', 'action_logprob', 'next_attention_map', 'reward', 'done', 'state_value', 'values'))


class Agent:
    def __init__(self, decision_model, task_embedding, eye, env, device, env_args, test_eye=None, test_env=None, memory=None) -> None:
        self.decision_model = decision_model.to(device)
        self.task_embedding = task_embedding.to(device)
        self.eye = eye
        self.env = env
        self.memory = memory
        self.device = device
        self.env_args = env_args
        self.test_eye = test_eye
        self.test_env = test_env

    def learn(self, args, train_args, logger):
        writer = SummaryWriter(args.logdir)
        for i_episode in range(args.num_episode):
            self.memory.clean()
            self.decision_model.eval()
            score, entropy = [], []

            # Explore
            observation, info = self.env.reset()
            self.eye.reset(info)
            for step in range(args.episode_step):
                similarity_maps, points, action, state_value, action_logprob, action_entropy, observation, reward, terminated, truncated, info = self.execute(
                    observation)
                entropy.append(action_entropy.item())
                self.memory.push(similarity_maps, action, action_logprob, None, reward,
                                 terminated or truncated or step == (args.episode_step-1), state_value, points)

                if terminated or truncated:
                    score.append(self.env.SCORE)
                    observation, info = self.env.reset()
                    self.eye.reset(info)
            print(len(score))
            print(i_episode, 'average_score', np.array(score).mean(),
                  'action entropy', np.array(entropy).mean())

            # Optimize
            summary = self.optimize(args, train_args)

            # Evaluate
            test_score = self.evaluate()

            # Save check point
            summary.update({'Score/explore': np.mean(score),
                           'Score/test': test_score})
            for key, value in summary.items():
                writer.add_scalar(key, value, i_episode)
            logger.write(summary)
            torch.save({
                'model_state_dict': self.decision_model.state_dict(),
                'embedding_model_state_dict': self.task_embedding.state_dict(),
                'logger': logger.summary
            }, args.checkpoint)

        self.env.close()

    def evaluate(self):
        observation, info = self.test_env.reset()
        self.test_eye.reset(info)

        for step in range(20):
            similarity_maps, points, action, state_value, action_logprob, action_entropy, observation, reward, terminated, truncated, info = self.execute(
                observation, True)
            if terminated or truncated:
                break
        score = self.test_env.SCORE
        return score

    def optimize(self, args, train_args):
        pi_loss_list, value_loss_list, entropy_loss_list, approx_kl_list = [], [], [], []
        self.memory.advantage_estimation()
        for i_epoch in range(args.num_epoch):
            actor_loss, critic_loss, entropy, approx_kl = self.train(
                train_args)
            pi_loss_list.append(actor_loss)
            value_loss_list.append(critic_loss)
            entropy_loss_list.append(entropy)
            approx_kl_list.append(approx_kl)
        summary = {
            'Loss/pi': np.mean(pi_loss_list),
            'Loss/v': np.mean(value_loss_list),
            'Loss/entropy': np.mean(entropy_loss_list),
            'Loss/kl': np.mean(approx_kl_list)
        }

        return summary

    def train(self, train_args):
        # self.decision_model.train()
        # self.task_embedding.train()

        if len(self.memory.memory) < train_args.BATCH_SIZE:
            return 0, 0, 0, 0
        transitions, advantages = self.memory.sample(train_args.BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        state_value = torch.tensor(batch.state_value).detach()

        advantages = torch.cat(advantages).unsqueeze(0)

        attention_map = torch.cat(batch.attention_map).to(self.device).detach()
        values = torch.cat(batch.values).detach()
        values = self.task_embedding(values.to(self.device))
        policy, value = self.decision_model(
            attention_map.to(self.device), values)
        policy = nn.functional.softmax(policy, 1)
        dist = Categorical(policy)

        action = torch.tensor(batch.action, device=self.device).unsqueeze(0)
        action = action.detach()
        action_logprobs = dist.log_prob(action)
        entropy = dist.entropy().mean()

        action_logprob_old = torch.tensor(
            batch.action_logprob, device=self.device)
        action_logprob_old = action_logprob_old.detach()

        ratios = torch.exp(action_logprobs.squeeze() - action_logprob_old)
        approx_kl = (action_logprob_old - action_logprobs).mean()
        advantages = advantages.detach().to(self.device)
        sur_1 = ratios * advantages
        sur_2 = torch.clamp(ratios, 1.0 - train_args.clip_eps,
                            1.0 + train_args.clip_eps) * advantages
        clip_loss = -torch.min(sur_1, sur_2)
        clip_loss = clip_loss.mean()

        actor_loss = clip_loss - train_args.entropy_coefficient * entropy

        target_values = state_value.to(self.device) + advantages

        value = value.squeeze()
        # target_values = (target_values - target_values.mean()
        #                  ) / target_values.std()

        critic_loss = train_args.criterion(value, target_values)
        # early stop
        if approx_kl > 0.015:
            loss = critic_loss
        else:
            loss = critic_loss + actor_loss

        # Optimize the model
        train_args.optimizer.zero_grad()
        train_args.embedding_optimizer.zero_grad()
        loss.backward()
        train_args.optimizer.step()
        train_args.embedding_optimizer.step()

        return clip_loss.item(), critic_loss.item(), entropy.item(), approx_kl.item()

    def evaluate_on_humanstimulus(self, env_args):
        observation, info = self.env.reset(target_image_file_list=env_args.target_image_file_list, distractor_image_file_list=env_args.distractor_image_file_list,
                                           distractor_index=env_args.distractor_index, all_sprited_positions=env_args.all_sprited_positions, values=env_args.points, popularities=env_args.popularity)
        self.eye.reset(info)

        for step in range(20):
            _, _, _, _, _, _, observation, _, terminated, truncated, info = self.execute(
                observation)
            if step == 19:
                truncated = True
            if terminated or truncated:
                print(self.env.SCORE / self.env.upperbound, step)
                break
        click_count = np.zeros((4, 20))
        for s, c in enumerate(info['click list']):
            if c == 'Target 1':
                click_count[0, s] = 1
            elif c == 'Target 2':
                click_count[1, s] = 1
            elif c == 'Target 3':
                click_count[2, s] = 1
            elif c == 'Target 4':
                click_count[3, s] = 1
        return {'score': self.env.SCORE / self.env.upperbound, 'click count': click_count}

    def execute(self, observation, is_test=False):
        similarity_maps, points = self.eye.visual_process(observation)
        attention_map, state_value = self.decision_model(
            similarity_maps, self.task_embedding(points))
        attention_map = nn.functional.softmax(attention_map, 1).squeeze()
        dist = Categorical(attention_map)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        action_entropy = dist.entropy()

        b, a = divmod(int(action.cpu()), self.env_args.size)
        if is_test:
            self.test_eye.fixations.append([a, b])
            next_observation, reward, terminated, truncated, info = self.test_env.step(np.array([
                a, b]))
        else:
            self.eye.fixations.append([a, b])
            next_observation, reward, terminated, truncated, info = self.env.step(np.array([
                a, b]))

        return similarity_maps.cpu(), points.cpu(), action.item(), state_value.item(), action_logprob.item(), action_entropy, next_observation, reward, terminated, truncated, info


class FixationAgent(Agent):
    def __init__(self, decision_model, eye, env, device, env_args, memory=None) -> None:
        self.decision_model = decision_model.to(device)
        self.eye = eye
        self.env = env
        self.memory = memory
        self.device = device
        self.env_args = env_args

    def train(self, train_args):
        # self.decision_model.train()
        if len(self.memory.memory) < train_args.BATCH_SIZE:
            return 0, 0, 0, 0
        transitions, advantages = self.memory.sample(
            train_args.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        state_value = torch.tensor(batch.state_value)
        state_value = state_value.detach().to(self.device)

        advantages = torch.cat(advantages).unsqueeze(0)
        attention_map = torch.cat(batch.attention_map)
        attention_map = attention_map.detach().to(self.device)
        points = torch.cat(batch.values)
        points = points.detach().to(self.device)
        policy, click_p, value = self.decision_model(attention_map, points)
        policy = nn.functional.softmax(policy, 1)
        dist = Categorical(policy)
        c_dist = Categorical(torch.cat((click_p, 1-click_p), 1).squeeze())

        action = torch.tensor(batch.action, device=self.device)
        action = action.detach().squeeze()
        click = action[:, 0]
        action = action[:, 1]
        action_logprobs = dist.log_prob(action.unsqueeze(
            0)) + c_dist.log_prob(click.unsqueeze(0))

        entropy = (dist.entropy()+c_dist.entropy()).mean()

        action_logprob_old = torch.tensor(
            batch.action_logprob, device=self.device)
        action_logprob_old = action_logprob_old.detach()

        ratios = torch.exp(action_logprobs.squeeze() - action_logprob_old)
        approx_kl = (action_logprob_old - action_logprobs).mean()
        advantages = advantages.detach().to(self.device)
        sur_1 = ratios * advantages
        sur_2 = torch.clamp(ratios, 1.0 - train_args.clip_eps,
                            1.0 + train_args.clip_eps) * advantages
        clip_loss = -torch.min(sur_1, sur_2)
        clip_loss = clip_loss.mean()

        actor_loss = clip_loss - train_args.entropy_coefficient * entropy

        target_values = state_value.to(self.device) + advantages

        value = value.squeeze()

        critic_loss = train_args.criterion(value, target_values)

        if approx_kl > 0.015:
            loss = critic_loss
        else:
            loss = critic_loss + actor_loss

        # Optimize the model
        train_args.optimizer.zero_grad()
        loss.backward()
        train_args.optimizer.step()

        return actor_loss.item(), critic_loss.item(), entropy.item(), approx_kl.item()

    def execute(self, observation, alwaysclick=False):
        similarity_maps, points = self.eye.visual_process(observation)
        attention_map, click_policy, state_value = self.decision_model(
            similarity_maps, points)
        attention_map = nn.functional.softmax(attention_map, 1)
        attention_map = attention_map.squeeze().cpu()
        attention_dist = Categorical(attention_map)
        click_dist = Categorical(
            torch.cat((click_policy, 1-click_policy)).squeeze())
        eyemovement = attention_dist.sample()
        click = click_dist.sample()
        action_logprob = attention_dist.log_prob(
            eyemovement) + click_dist.log_prob(click)
        action_entropy = attention_dist.entropy() + click_dist.entropy()

        b, a = divmod(int(eyemovement.item()), self.env_args.size)
        self.eye.fixations.append([a, b])
        action = [click.item(), a, b]
        if alwaysclick:
            action[0] = 1
        next_observation, reward, terminated, truncated, info = self.env.step(
            np.array(action))
        return similarity_maps.cpu(), points.cpu(), [click.item(), eyemovement.item()], state_value.item(), action_logprob.item(), action_entropy, next_observation, reward, terminated, truncated,

    def execute_baseline_1(self, observation):
        # popularity first
        similarity_maps, points = self.eye.visual_process(observation)
        attention_maps = similarity_maps.squeeze()
        attention_map = attention_maps.view(-1)
        policy = nn.functional.softmax(attention_map / 0.1, -1)
        # action = torch.argmax(attention_map).cpu()
        dist = Categorical(policy)
        action = dist.sample().cpu()

        z = action // (attention_maps.size(1) * attention_maps.size(2))
        y = (
            action % (attention_maps.size(1) * attention_maps.size(2))
        ) // attention_maps.size(2)
        x = (
            action % (attention_maps.size(1) * attention_maps.size(2))
        ) % attention_maps.size(2)
        self.eye.fixations.append([x, y])
        action = [1, x, y]

        next_observation, reward, terminated, truncated, info = self.env.step(
            np.array(action))

        return action, next_observation, reward, terminated, truncated, info

    def execute_baseline_2(self, observation):
        # value first
        similarity_maps, points = self.eye.visual_process(observation)
        attention_maps = (similarity_maps*points.view(-1, 4, 1, 1)).squeeze()
        attention_map = attention_maps.view(-1)
        policy = nn.functional.softmax(attention_map / 0.1, -1)
        # action = torch.argmax(attention_map).cpu()
        dist = Categorical(policy)
        action = dist.sample().cpu()

        z = action // (attention_maps.size(1) * attention_maps.size(2))
        y = (
            action % (attention_maps.size(1) * attention_maps.size(2))
        ) // attention_maps.size(2)
        x = (
            action % (attention_maps.size(1) * attention_maps.size(2))
        ) % attention_maps.size(2)
        self.eye.fixations.append([x, y])
        action = [1, x, y]

        next_observation, reward, terminated, truncated, info = self.env.step(
            np.array(action))

        return action, next_observation, reward, terminated, truncated, info

    def execute_baseline_3(self, observation):
        # weighted sum
        similarity_maps, points = self.eye.visual_process(observation)

        attention_map = similarity_maps * points.view(-1, 4, 1, 1)
        attention_map = torch.sum(attention_map, 1)
        attention_map = attention_map.view(-1)
        policy = nn.functional.softmax(attention_map / 0.1, -1)
        # action = torch.argmax(attention_map).cpu()
        dist = Categorical(policy)
        action = dist.sample()
        y, x = divmod(int(action.cpu()), self.env_args.size)
        self.eye.fixations.append([x, y])
        action = [1, x, y]

        next_observation, reward, terminated, truncated, info = self.env.step(
            np.array(action))

        return action, next_observation, reward, terminated, truncated, info

    def execute_baseline_4(self, observation):
        # chance: random select target
        action_set = self.env.target_ones + self.env.target_twos + \
            self.env.target_threes + self.env.target_fours
        fixation = random.choice(action_set)
        action = [1, fixation[0], fixation[1]]

        next_observation, reward, terminated, truncated, info = self.env.step(
            np.array(action))

        return action, next_observation, reward, terminated, truncated, info
    
    def execute_pure_random(self):

        action = [1, self.env.action_space.sample(), self.env.action_space.sample()]

        next_observation, reward, terminated, truncated, info = self.env.step(
            np.array(action))

        return action, next_observation, reward, terminated, truncated, info

    def evaluate(self):
        observation, info = self.env.reset()
        self.eye.reset(info)

        clicked = 1e-4
        correct_click = 0

        for step in range(80):
            similarity_maps, points, action, state_value, action_logprob, action_entropy, observation, reward, terminated, truncated, info = self.execute(
                observation)
            if reward > 0 and action[0]:
                correct_click += 1
            if terminated or truncated:
                break
        score = self.env.SCORE
        clicked += self.env.clicked
        print('step:', step, 'click:', clicked, 'score',
              score, 'correct rate:', correct_click/clicked)
        return score

    def evaluate_on_humanstimulus(self, env_args, alwaysclick=False, baseline=None):
        observation, info = self.env.reset(target_image_file_list=env_args.target_image_file_list, distractor_image_file_list=env_args.distractor_image_file_list,
                                           distractor_index=env_args.distractor_index, all_sprited_positions=env_args.all_sprited_positions, values=env_args.points, popularities=env_args.popularity)
        self.eye.reset(info)
        self.decision_model.eval()

        clicked = 0
        correct_click = 0
        cumulative_score = []
        click_positions = []
        radius_score = []
        for step in range(200):
            if baseline == 'value':
                action, observation, reward, terminated, truncated, info = self.execute_baseline_2(
                    observation)
            elif baseline == 'popularity':
                action, observation, reward, terminated, truncated, info = self.execute_baseline_1(
                    observation)
            elif baseline == 'add':
                action, observation, reward, terminated, truncated, info = self.execute_baseline_3(
                    observation)
            elif baseline == 'chance':
                action, observation, reward, terminated, truncated, info = self.execute_baseline_4(
                    observation)
            elif baseline == 'pure chance':
                action, observation, reward, terminated, truncated, info = self.execute_pure_random()
            else:
                _, _, action, _, _, _, observation, reward, terminated, truncated, info = self.execute(
                    observation, alwaysclick)
            clicked += action[0]
            radius_score.append(info['radius score'])
            if action[0]:
                cumulative_score.append(self.env.SCORE)
                click_positions.append(self.env.fixations[-1])
            if action[0] and reward > 0:
                correct_click += 1
            if terminated or truncated:
                print(self.env.SCORE / self.env.upperbound, step)
                break

        item_percentage = [info['click list'].count("Target 1"), info['click list'].count("Target 2"), info['click list'].count(
            "Target 3"), info['click list'].count("Target 4"), info['click list'].count("Distractor"), info['click list'].count("blank")]
        click_count = np.zeros((4, 20))
        for s, c in enumerate(info['click list']):
            if c == 'Target 1':
                click_count[0, s] = 1
            elif c == 'Target 2':
                click_count[1, s] = 1
            elif c == 'Target 3':
                click_count[2, s] = 1
            elif c == 'Target 4':
                click_count[3, s] = 1
        result = {'score': self.env.SCORE / self.env.upperbound,
                  'click ratio': clicked / (step+1),
                  'item percentage': np.array(item_percentage),
                  'click count': click_count,
                  'correct click rate': correct_click / (clicked+1e-4),
                  'fixation positions': np.array(self.env.fixations),
                  'click positions': np.array(click_positions),
                  'cumulative score': np.array(cumulative_score) / self.env.upperbound,
                  'radius score': radius_score
                  }
        return result


class AnotherFixationAgent(FixationAgent):
    def __init__(self, decision_model, task_embedding, eye, env, device, env_args, memory=None) -> None:
        self.decision_model = decision_model.to(device)
        self.task_embedding = task_embedding.to(device)
        self.eye = eye
        self.env = env
        self.memory = memory
        self.device = device
        self.env_args = env_args

    def execute(self, observation, alwaysclick=False):
        similarity_maps, points = self.eye.visual_process(observation)
        attention_map, click_policy, state_value = self.decision_model(
            similarity_maps, self.task_embedding(points))
        attention_map = nn.functional.softmax(attention_map, 1)
        attention_map = attention_map.squeeze().cpu()
        attention_dist = Categorical(attention_map)
        click_dist = Categorical(
            torch.cat((click_policy, 1-click_policy)).squeeze())
        eyemovement = attention_dist.sample()
        click = click_dist.sample()
        action_logprob = attention_dist.log_prob(
            eyemovement) + click_dist.log_prob(click)
        action_entropy = attention_dist.entropy() + click_dist.entropy()

        b, a = divmod(int(eyemovement.item()), self.env_args.size)
        self.eye.fixations.append([a, b])
        action = [click.item(), a, b]
        if alwaysclick:
            action[0] = 1
        next_observation, reward, terminated, truncated, info = self.env.step(
            np.array(action))
        return similarity_maps.cpu(), points.cpu(), [action[0], eyemovement.item()], state_value.item(), action_logprob.item(), action_entropy, next_observation, reward, terminated, truncated, info
