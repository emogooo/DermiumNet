import numpy as np


class EquilibriumOptimizer:
    def _init_(self, obj_func, bounds, num_agents, max_iter):
        self.obj_func = obj_func
        self.bounds = bounds
        self.num_agents = num_agents
        self.max_iter = max_iter

    def optimize(self):
        # Initialization
        dim = len(self.bounds)  # Bounds = Yüksek İhtimal 2 Boyutlu Dizi
        agents = np.random.rand(self.num_agents, dim)
        agents = agents * (self.bounds[:, 1] - self.bounds[:, 0]) + self.bounds[:, 0]
        best_agent = agents[0]
        best_fitness = float("inf")

        for iteration in range(self.max_iter):
            # Evaluate fitness
            fitness = np.array([self.obj_func(agent) for agent in agents])  # ahanda iş burada bitiyi
            if np.min(fitness) < best_fitness:
                best_fitness = np.min(fitness)
                best_agent = agents[np.argmin(fitness)]

            # Update agents
            for i in range(self.num_agents):
                agent = agents[i]
                a = 2 * (1 - iteration / self.max_iter)
                r1 = np.random.rand()
                r2 = np.random.rand()
                g = np.random.rand()

                if g < 0.5:
                    agent = agent + a * r1 * (best_agent - r2 * agent)
                else:
                    agent = agent - a * r1 * (best_agent - r2 * agent)

                # Boundary check
                agents[i] = np.clip(agent, self.bounds[:, 0], self.bounds[:, 1])

        return best_agent, best_fitness
