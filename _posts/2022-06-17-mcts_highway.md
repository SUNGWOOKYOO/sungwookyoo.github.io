---
title: "MCTS for Autonomous Driving"
excerpt: "study"
categories:
 - study
tags:
 - ml
use_math: true
last_modified_at: "2022-06-17"
toc: true
toc_sticky: true
toc_label: "On this page"
toc_icon: "cog"
header:
 overlay_image: /assets/images/teaser.jpg
 overlay_filter: 0.5
 caption: mcts를 실제 문제 어플리케이션에 시뮬레이션에 적용해보자.
 actions:
  - label: "#"
    url: "#"
---

# MCTS 를  자율주행에 적용

MCTS를 자율주행과 같은 실제 문제에 적용해보면 

어느정도의 성능이 나오고 한계점이 무엇인지 직접 확인해보기 위해서  

[rl-agents](https://github.com/eleurent/rl-agents) 라는 오픈 소스를 사용해서 실제 돌려보고 분석해봤다.



## MCTS Agent 의 생성



### Agent 상속구조

우선 MCTS Agent의 상속구조부터 살펴보자.

Configurable, ABC - AbstractAgent - AbstractTreeSearchAgent - MCTSAgent 로 구성된다.

MCTSAgent를 생성할 때 Configurable 객체에 default_config 가 class method 데코레이터를 사용하여 껍데기를 정의해 두었다. 

결과적으로는 AbstractTreeSearchAgent 에  overriding된 default_config를 불러와서 agent를 초기화한다. 

그리고 추가적인 매개변수에 대해서 rec_update 메서드를 사용하여 config를 업데이트 한다.

### Agent 생성자

처음에 MCTS Agent를 다음과 같이 생성하면 

```python
agent = MCTSAgent(env, config={"env_preprocessors": [{"method":"simplify"}],
                                   "budget": 75,
                                   "gamma": 0.7})
```

이곳부터 호출이 된다.

Config가 많기 때문에 모든 값의 의미를 일일히 알 필요는 없다. 실행을 위해서 필요한 값의 의미를 Top-down식으로 살펴보자.

```python
class AbstractTreeSearchAgent(AbstractAgent):
    PLANNER_TYPE = None
    NODE_TYPE = None

    def __init__(self,
                 env,
                 config=None):
        """
            A new Tree Search agent.
        :param env: The environment
        :param config: The agent configuration. Use default if None.
        """
        super(AbstractTreeSearchAgent, self).__init__(config)
        self.env = env
        self.planner = self.make_planner()
        self.previous_actions = []
        self.remaining_horizon = 0
        self.steps = 0

```

그리고 상속을 따라 쭉 호출되서 마지막에 이곳이 호출된다.

```python
class Configurable(object):
    """
        This class is a container for a configuration dictionary.
        It allows to provide a default_config function with pre-filled configuration.
        When provided with an input configuration, the default one will recursively be updated,
        and the input configuration will also be updated with the resulting configuration.
    """
    def __init__(self, config=None):
        self.config = self.default_config()
        if config:
            # Override default config with variant
            Configurable.rec_update(self.config, config)
            # Override incomplete variant with completed variant
            Configurable.rec_update(config, self.config)
```

default_config는 MCTSAgent 클래스에 다음과 같이 정의되어 있다.

```python
class MCTSAgent(AbstractTreeSearchAgent):
    """
        An agent that uses Monte Carlo Tree Search to plan a sequence of action in an MDP.
    """
    def make_planner(self):
        prior_policy = MCTSAgent.policy_factory(self.config["prior_policy"])
        rollout_policy = MCTSAgent.policy_factory(self.config["rollout_policy"])
        return MCTS(self.env, prior_policy, rollout_policy, self.config)
    
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            "budget": 100,
            "horizon": None,
            "prior_policy": {"type": "random_available"},
            "rollout_policy": {"type": "random_available"},
            "env_preprocessors": []
         })
        return config
```

### Planer 생성자

Agent 생성자에서 planner 인스턴스를 매서드 변수로 할당한다. 

Planner가 MCTS로 동작하게 되며 시뮬레이션 env, prior_policy, rollout_policy, config를 모두 매개변수로 입력받는다.

prior와 rollout policy는 "random_available" 라는 key를 갖는 policy로 생성된다.

```python
class MCTSAgent(AbstractTreeSearchAgent):
    ...
	@staticmethod
    def policy_factory(policy_config):
        if policy_config["type"] == "random":
            return MCTSAgent.random_policy
        elif policy_config["type"] == "random_available":
            return MCTSAgent.random_available_policy
        elif policy_config["type"] == "preference":
            return partial(MCTSAgent.preference_policy,
                           action_index=policy_config["action"],
                           ratio=policy_config["ratio"])
        else:
            raise ValueError("Unknown policy type")
```

random available policy는 현재 가능한 action들 중 하나를 uniform하게 뽑는 policy이다.

```python
	@staticmethod
    def random_policy(state, observation):
        """
            Choose actions from a uniform distribution.

        :param state: the environment state
        :param observation: the corresponding observation
        :return: a tuple containing the actions and their probabilities
        """
        actions = np.arange(state.action_space.n)
        probabilities = np.ones((len(actions))) / len(actions)
        return actions, probabilities
    
    @staticmethod
    def random_available_policy(state, observation):
        """
            Choose actions from a uniform distribution over currently available actions only.

        :param state: the environment state
        :param observation: the corresponding observation
        :return: a tuple containing the actions and their probabilities
        """
        if hasattr(state, 'get_available_actions'):
            available_actions = state.get_available_actions()
        else:
            available_actions = np.arange(state.action_space.n)
        probabilities = np.ones((len(available_actions))) / len(available_actions)
        return available_actions, probabilities
```

Planner의 상속관계는 다음과 같다.

Configurable - AbstractPlanner - MCTS

MCTS를 호출하면 from rl_agents.agents.tree_search.olop 에 정의된 static method를 사용하여 

시뮬레이션을 몇번돌릴지 (episodes) 어느정도 깊이까지 tree를 만들지 (horizon) 를 budget과 gamma를 가지고 계산한다.

```python
class MCTS(AbstractPlanner):
    """
       An implementation of Monte-Carlo Tree Search, with Upper Confidence Tree exploration.
    """
    def __init__(self, env, prior_policy, rollout_policy, config=None):
        """
            New MCTS instance.

        :param config: the mcts configuration. Use default if None.
        :param prior_policy: the prior policy used when expanding and selecting nodes
        :param rollout_policy: the rollout policy used to estimate the value of a leaf node
        """
        super().__init__(config)
        self.env = env
        self.prior_policy = prior_policy
        self.rollout_policy = rollout_policy
        if not self.config["horizon"]:
            self.config["episodes"], self.config["horizon"] = \
                OLOP.allocation(self.config["budget"], self.config["gamma"])
                
    class OLOP(AbstractPlanner):
    """
       An implementation of Open Loop Optimistic Planning.
    """
    ...
    @staticmethod
    def allocation(budget, gamma):
        """
            Allocate the computational budget into M episodes of fixed horizon L.
        """
        for episodes in range(1, int(budget)):
            if episodes * OLOP.horizon(episodes, gamma) > budget:
                episodes = max(episodes - 1, 1)
                horizon = OLOP.horizon(episodes, gamma)
                break
        else:
            raise ValueError("Could not split budget {} with gamma {}".format(budget, gamma))
        return episodes, horizon
    
 	@staticmethod
    def horizon(episodes, gamma):
        return max(int(np.ceil(np.log(episodes) / (2 * np.log(1 / gamma)))), 1)
```

MCTS를 위한 시뮬레이션의 너비와 깊이에 대한 hyper parameter를 찾는데 

휴리스틱한 식을 통해서 구하는 것 같은데 왜 이렇게 하는지는 정확히 모르겠다.

오픈소스에 나온대로 값을 초기화하면 episodes 는 17, horizon 4 로 계산된다.

또한 MCTS planner를 생성할때 Agent때와 마찬가지로 MCTS  의 default config가 호출된다.

```python
	@classmethod
    def default_config(cls):
        cfg = super(MCTS, cls).default_config()
        cfg.update({
            "temperature": 2 / (1 - cfg["gamma"]),
            "closed_loop": False
        })
        return cfg
```

MCTS parent의 default config 도 호출해 반드시 필요한 config를  초기화한다.

```python
class AbstractPlanner(Configurable):
    def __init__(self, config=None):
        super().__init__(config)
        self.np_random = None
        self.root = None
        self.observations = []
        self.reset()
        self.seed()

    @classmethod
    def default_config(cls):
        return dict(budget=500,
                    gamma=0.8,
                    step_strategy="reset")
```



### Planner 의 실행

MCTS 에는 act가 정의되어 있지 않다 그래서 그 부모 클래스로 가면 act가 정의되어있다.

agent.act를 실행하면 plan함수를 호출하고 plan된 값중 현재 스텝에 필요한 0번째 action을  반환한다.

```python
action = agent.act(state)

class AbstractTreeSearchAgent(AbstractAgent):
    def act(self, state):
        return self.plan(state)[0]

    def plan(self, observation):
        """
            Plan an optimal sequence of actions.

            Start by updating the previously found tree with the last action performed.

        :param observation: the current state
        :return: the list of actions
        """
        self.steps += 1
        replanning_required = self.step(self.previous_actions)
        if replanning_required:
            env = preprocess_env(self.env, self.config["env_preprocessors"])
            actions = self.planner.plan(state=env, observation=observation)
        else:
            actions = self.previous_actions[1:]
        self.write_tree()

        self.previous_actions = actions
        return actions
```


plan함수는 AbstractTreeSearchAgent 와 MCTS 에 overloading 이 되어있다는 것에 주의하자.

AbstractTreeSearchAgent.plan 은 observation 만 받을 때 호출되고

MCTS.plan은 state=env와 observation 두 변수를 받을 때 호출된다.

우선 전자가 먼저 호출된다.

AbstractTreeSearchAgent.plan 에서는 replanning 여부를 체크하고  필요한 경우에만 MCTS.plan를 호출한다.

replanning 여부를 확인하는 step함수로 들어가보자.

remaining_horizon 이 없을 때 replaning을 하게 된다.

```python
class AbstractTreeSearchAgent(AbstractAgent):	
    def step(self, actions):
        """
            Handle receding horizon mechanism
        :return: whether a replanning is required
        """
        replanning_required = self.remaining_horizon == 0 or len(actions) <= 1
        if replanning_required:
            self.remaining_horizon = self.config["receding_horizon"] - 1
        else:
            self.remaining_horizon -= 1

        self.planner.step_tree(actions)
        return replanning_required
```
이후에 planner의 step_tree함수에서 planner tree 업데이트한다.

step_strategy 의 파라미터가 무엇인지에 따라서 업데이트 방식이 다르다.

AbstractPlanner.default_config 에 의해서 step_strategy의 key는 reset 임을 알 수 있다.

```python
	def step_tree(self, actions):
        """
            Update the planner tree when the agent performs an action

        :param actions: a sequence of actions to follow from the root node
        """
        if self.config["step_strategy"] == "reset":
            self.step_by_reset()
        elif self.config["step_strategy"] == "subtree":
            if actions:
                self.step_by_subtree(actions[0])
            else:
                self.step_by_reset()
        else:
            logger.warning("Unknown step strategy: {}".format(self.config["step_strategy"]))
            self.step_by_reset()
```

AbstractPlanner에 정의된 step_by_reset 과 step_by_subtree 는 다음과 같다.

step_by_reset 에서는

self.reset() 을 호출할 때는 이를 상속받은 MCTS의 reset이 호출되므로 root node를 초기화한다.  

step_by_subtree 에서는

선택된 action에 대한 노드의 subtree를 None으로 할당하여 해당 노드를 Leaf node로 만든다.



~~~python
class AbstractPlanner(Configurable):
    def step_by_reset(self):
        """
            Reset the planner tree to a root node for the new state.
        """
        self.reset()

    def step_by_subtree(self, action):
        """
            Replace the planner tree by its subtree corresponding to the chosen action.

        :param action: a chosen action from the root node
        """
        if action in self.root.children:
            self.root = self.root.children[action]
            self.root.parent = None
        else:
            # The selected action was never explored, start a new tree.
            self.step_by_reset()
class AbstractPlanner(Configurable):
    def reset(self):
        raise NotImplementedError
        
class MCTS(AbstractPlanner):
    ```
    def reset(self):
        self.root = MCTSNode(parent=None, planner=self)
~~~

요약하자면 현재의 step stretegy 따르면 매스텝 root 노드를 초기화하고 정해진 horizon마다 planning을 하고 이를 유지하는 것이다.

다시 돌아가서 정해진 horizon마다 오버로딩된 self.planner.plan 속에서 어떻게 plannning을 하는지 보자. 

시뮬레이션 돌릴 수 만큼 환경을 깊은복사를 하고 run함수를 통해서 시뮬레이션을 돌린다.

시뮬레이션을 분기하는 방법은 Leaf, Root, Leaf와 Root복합 인데 위 코드에서 구현한 것은  Root에서 분기를 했다.

여기서 효과적으로 하려면 run을 멀리 쓰레드를 사용하여 병렬로 돌리면 된다.

이후 exploitation 을 위한 get_plan함수를 호출한다.

```python
class MCTS(AbstractPlanner):
	def plan(self, state, observation):
        for i in range(self.config['episodes']):
            if (i+1) % 10 == 0:
                logger.debug('{} / {}'.format(i+1, self.config['episodes']))
            self.run(safe_deepcopy_env(state), observation)
        return self.get_plan()
```

run함수로 들어가 어떻게 시뮬레이션을 하는 지 살펴보자.

이 부분이 **MCTS의 핵심부분**이다.

```python
class MCTS(AbstractPlanner):	
    def run(self, state, observation):
        """
            Run an iteration of Monte-Carlo Tree Search, starting from a given state

        :param state: the initial environment state
        :param observation: the corresponding observation
        """
        node = self.root
        total_reward = 0
        depth = 0
        terminal = False
        state.seed(self.np_random.randint(2**30))
        while depth < self.config['horizon'] and node.children and not terminal:
            action = node.sampling_rule(temperature=self.config['temperature'])
            observation, reward, terminal, _ = self.step(state, action)
            total_reward += self.config["gamma"] ** depth * reward
            node_observation = observation if self.config["closed_loop"] else None
            node = node.get_child(action, observation=node_observation)
            depth += 1

        if not node.children \
                and depth < self.config['horizon'] \
                and (not terminal or node == self.root):
            node.expand(self.prior_policy(state, observation))

        if not terminal:
            total_reward = self.evaluate(state, observation, total_reward, depth=depth)
        node.update_branch(total_reward)
```

#### Selection

우선 root 노드에서 Leaf 노드에 도달할 때까지 아래와 같은 sampling rule에 따라서 action을 선택한다.

childen중에서 현재까지 모은 정보를 바탕으로 Temperature에 따라서 exploration 과 exploitation을 고려하여 Selection과정을 한다.

만약 Leaf 노드라면 None을 반환한다.

```python
class MCTSNode(Node):
    def sampling_rule(self, temperature=None):
        """
            Select an action from the node.
            - if exploration is wanted with some temperature, follow the selection strategy.
            - else, select the action with maximum visit count

        :param temperature: the exploration parameter, positive or zero
        :return: the selected action
        """
        if self.children:
            actions = list(self.children.keys())
            # Randomly tie best candidates with respect to selection strategy
            indexes = [self.children[a].selection_strategy(temperature) for a in actions]
            return actions[self.random_argmax(indexes)]
        else:
            return None

    def selection_strategy(self, temperature):
        """
            Select an action according to its value, prior probability and visit count.

        :param temperature: the exploration parameter, positive or zero.
        :return: the selected action with maximum value and exploration bonus.
        """
        if not self.parent:
            return self.get_value()

        # return self.value + temperature * self.prior * np.sqrt(np.log(self.parent.count) / self.count)
        return self.get_value() + temperature * len(self.parent.children) * self.prior/(self.count+1)

    def get_value(self):
        return self.value
```
이후 현 상태에서 선택된 action에 대해서 step 함수를 호출하는데 MCTS 는 정의되어있지 않기 때문에 AbstractPlanner의 step을 호출한다.

```python
class AbstractPlanner(Configurable):
	def step(self, state, action):
        observation, reward, done, info = state.step(action)
        self.observations.append(observation)
        return observation, reward, done, info
```

state = 깊은 복사가 이루어진 environment의 인스턴스 step함수를 호출하여 leaf노드까지 시뮬레이션을 진행한다. 

select을 통해 얻은 정보들을 total_reward에 합산한다.

```python
def get_child(self, action, observation=None):
        child = self.children[action]
        if observation is not None:
            if str(observation) not in child.children:
                child.children[str(observation)] = MCTSNode(parent=child, planner=self.planner, prior=0)
            child = child.children[str(observation)]
        return child
```

다음 노드로 갈때 get_child 함수를 사용하여 해당 observation와 action에 대한 child 노드를 얻는다.

여기서 이상한 점은 모든 action에 대해서 child가 없을 수 도 있지 않냐고 생각할 수 있다. 

이 코드에서는 node를 expansion할 때 모든 action에 대해서 항상 노드를 만들기 때문에 그러한 경우는 존재하지 않는다. 

만약 action 차원이 높거나 continous action일 경우 이 부분에 수정이 필요할 것이다.

그리고 default parameter로 closed_loop는 False 이기 때문에 observation 은 child 노드에 고려를 하지 않는다. 

사실 이것도 고려해야 더 정밀한 tree가 될 것이다. 하지만 overhead때문에 안하는 것 같다.

#### Expansion

maximum horizon 에 도달하지 않은 leaf 노드라면 prior_policy 를 사용하여 expansion을 한다.

prior_policy는 위에서 random policy로 정의하였다.

```python
		if not node.children \
                and depth < self.config['horizon'] \
                and (not terminal or node == self.root):
            node.expand(self.prior_policy(state, observation))
```

leaf 노드를 expand하는 함수는 다음과 같다.

parent를 자기자신 self 로 하고 planner 인스턴스는 공유하고  분기되는 prior는 uniform 확률로 하여 자식노드를 초기화 한다.

```python
	def expand(self, actions_distribution):
        """
            Expand a leaf node by creating a new child for each available action.

        :param actions_distribution: the list of available actions and their prior probabilities
        """
        actions, probabilities = actions_distribution
        for i in range(len(actions)):
            if actions[i] not in self.children:
                self.children[actions[i]] = type(self)(self, self.planner, probabilities[i])
```



#### Simulation

terminal state가 아니라면 evaluation 함수를 호출하여 simulation을 진행하고 tree를 업데이트하기 위한 reward를 받아온다.

위에서 rollout_policy를 random_available_policy 로 정의했다.  

모든 thread에 대해서 주어진 horizon 까지 임의로 하나의 행동을 골라 환경에 이행하고

얻어진 cumulative sum of reward를 반환한다.

```python
class MCTS(AbstractPlanner):   
    def evaluate(self, state, observation, total_reward=0, depth=0):
        """
            Run the rollout policy to yield a sample of the value of being in a given state.
    
        :param state: the leaf state.
        :param observation: the corresponding observation.
        :param total_reward: the initial total reward accumulated until now
        :param depth: the initial simulation depth
        :return: the total reward of the rollout trajectory
        """
        for h in range(depth, self.config["horizon"]):
            actions, probabilities = self.rollout_policy(state, observation)
            action = self.np_random.choice(actions, 1, p=np.array(probabilities))[0]
            observation, reward, terminal, _ = self.step(state, action)
            total_reward += self.config["gamma"] ** h * reward
            if np.all(terminal):
                break
        return total_reward
```

#### Back Propagation

마지막으로 시뮬레이션을 통해 얻은 정보를 사용해 tree의 노드들의 정보를 back propagation해야한다.

update_branch 함수로 들어가서 살펴보자.

```python
	def update_branch(self, total_reward):
        """
            Update the whole branch from this node to the root with the total reward of the corresponding trajectory.

        :param total_reward: the total reward obtained through a trajectory passing by this node
        """
        self.update(total_reward)
        if self.parent:
            self.parent.update_branch(total_reward)
```

먼저 자신의 노드의 count와 value를 업데이트하고  

부모 노드를 update_branch 함수를 호출하여 업데이트한다.

update 함수는 다음과 같다.

```python
	def update(self, total_reward):
        """
            Update the visit count and value of this node, given a sample of total reward.

        :param total_reward: the total reward obtained through a trajectory passing by this node
        """
        self.count += 1
        self.value += self.K / self.count * (total_reward - self.value)
```

#### Exploitation

search tree를 업데이트하고 나면 plan함수에선 다음과 같은 get_plan을 호출하여 값을 반환한다.

root 노드에서 부터 children이 없는 leaf 노드까지 

selection_rule 함수를 호출하여 optimal action sequence를 얻고 이를 반환한다.

```python
	def get_plan(self):
        """
            Get the optimal action sequence of the current tree by recursively selecting the best action within each
            node with no exploration.

        :return: the list of actions
        """
        actions = []
        node = self.root
        while node.children:
            action = node.selection_rule()
            actions.append(action)
            node = node.children[action]
        return actions
```

selection_rule 함수에서는 sampling_rule과 달리 temperature가 곱해진 항이 없이 exploitation만 고려하여 best action을 뽑는다.

children들의 시뮬레이션에서 많이 방문했고 동시에 value가 가장 높았던 action을 고르게 된다.

```python
class MCTSNode(Node):	
    def selection_rule(self):
        if not self.children:
            return None
        # Tie best counts by best value
        actions = list(self.children.keys())
        counts = Node.all_argmax([self.children[a].count for a in actions])
        return actions[max(counts, key=(lambda i: self.children[actions[i]].get_value()))]
    
class Node(object):
    """
        A tree node
    """   
    def all_argmax(x):
        """
        :param x: a set
        :return: the list of indexes of all maximums of x
        """
        m = np.amax(x)
        return np.nonzero(x == m)[0]
```



## 실행결과

![result](/assets/images/mcts_highway_files/highway_mcts_agent.gif)

agent 의 1step computation time : 5.45s

simulation 의 1 fame 걸리는 시간 : 0.67s

꾀 좋은 결과를 보이지만,

시뮬레이션에 걸리는 시간에 따라서

agent의 연산시간이 크게 증가하여 실제 적용에는 한계를 보인다.



## 고찰

Decision Maker로  MCTS를 통해서 활용한 사례였다.

자율주행은 기본적으로 에피소드가 끝나기 까지 긴시간이 걸리기 때문에

 search tree의 깊이를 hyper parameter로 정해두고 문제를 접근했다.

한가지 의문점은 구현된 코드에서는 receding horizon마다 root node를 초기화한다는 것이다.

그렇게 되면 에피소드 중간에 search tree 학습된 정보를 잃어버리기 때문에 무지성 행동을 하게 될 것이다.

중간중간 초기화를 해주면 local optima로 빠지는 것을 막을 수 있기 때문에 그렇게 한 것 같다.

실제로도 그렇게 해야 더 동작을 잘한다는 것을 알았다. 

실제 문제에 적용하기 가장큰 문제점은 시뮬레이션이 필요하고 시간이 오래걸린다는 것이다.

시뮬레이션을 횟수를 늘리고 horizon을 늘리면 더 정확한 결과가 나올 수 있지만 

연산 시간상의 이유로 너무 높일 수 없기 때문에 한계가 발생하게 된다.



## 참조문헌

1. agent 코드 : [rl-agents](https://github.com/eleurent/rl-agents)

2. simulation : [highway](https://github.com/eleurent/highway-env)
