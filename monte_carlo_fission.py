import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random

class NeutronChainReaction:
    def __init__(self, 
                 fission_probability=0.8,
                 absorption_probability=0.15,
                 leakage_probability=0.05,
                 neutrons_per_fission_mean=2.4,
                 neutrons_per_fission_std=0.3):
        """
        Monte Carlo simulation of nuclear chain reaction
        
        Parameters:
        - fission_probability: Probability a neutron causes fission
        - absorption_probability: Probability a neutron is absorbed without fission
        - leakage_probability: Probability a neutron escapes the system
        - neutrons_per_fission_mean: Average neutrons released per fission
        - neutrons_per_fission_std: Standard deviation of neutrons per fission
        """
        self.p_fission = fission_probability
        self.p_absorption = absorption_probability
        self.p_leakage = leakage_probability
        
        # Ensure probabilities sum to 1
        total_p = self.p_fission + self.p_absorption + self.p_leakage
        if abs(total_p - 1.0) > 1e-6:
            print(f"Warning: Probabilities sum to {total_p}, normalizing...")
            self.p_fission /= total_p
            self.p_absorption /= total_p
            self.p_leakage /= total_p
        
        self.nu_mean = neutrons_per_fission_mean
        self.nu_std = neutrons_per_fission_std
        
        # Statistics tracking
        self.reset_statistics()
    
    def reset_statistics(self):
        """Reset simulation statistics"""
        self.total_fissions = 0
        self.total_absorptions = 0
        self.total_leakages = 0
        self.generation_sizes = []
        self.k_effective_history = []
    
    def sample_neutron_fate(self):
        """Sample what happens to a neutron"""
        rand = random.random()
        
        if rand < self.p_fission:
            return "fission"
        elif rand < self.p_fission + self.p_absorption:
            return "absorption"
        else:
            return "leakage"
    
    def sample_fission_neutrons(self):
        """Sample number of neutrons produced in fission"""
        # Use normal distribution, but ensure positive integer
        nu = np.random.normal(self.nu_mean, self.nu_std)
        return max(1, int(round(nu)))
    
    def simulate_generation(self, neutrons_in_generation):
        """Simulate one generation of neutron interactions"""
        next_generation_neutrons = 0
        fissions_this_gen = 0
        absorptions_this_gen = 0
        leakages_this_gen = 0
        
        for _ in range(neutrons_in_generation):
            fate = self.sample_neutron_fate()
            
            if fate == "fission":
                fissions_this_gen += 1
                # Sample number of neutrons produced
                new_neutrons = self.sample_fission_neutrons()
                next_generation_neutrons += new_neutrons
                
            elif fate == "absorption":
                absorptions_this_gen += 1
                
            else:  # leakage
                leakages_this_gen += 1
        
        # Update statistics
        self.total_fissions += fissions_this_gen
        self.total_absorptions += absorptions_this_gen
        self.total_leakages += leakages_this_gen
        
        return next_generation_neutrons, fissions_this_gen, absorptions_this_gen, leakages_this_gen
    
    def run_simulation(self, initial_neutrons=1, max_generations=50, min_neutrons=1):
        """
        Run complete chain reaction simulation
        
        Parameters:
        - initial_neutrons: Number of neutrons to start with
        - max_generations: Maximum generations to simulate
        - min_neutrons: Stop if neutron population drops below this
        
        Returns:
        - Dictionary with simulation results
        """
        self.reset_statistics()
        
        current_neutrons = initial_neutrons
        generation = 0
        
        generation_data = []
        
        print(f"Starting simulation with {initial_neutrons} neutron(s)")
        print("Gen | Neutrons | Fissions | Absorptions | Leakages | k_eff")
        print("-" * 65)
        
        while generation < max_generations and current_neutrons >= min_neutrons:
            next_neutrons, fissions, absorptions, leakages = self.simulate_generation(current_neutrons)
            
            # Calculate k-effective for this generation
            if current_neutrons > 0:
                k_eff = next_neutrons / current_neutrons
            else:
                k_eff = 0
            
            self.generation_sizes.append(current_neutrons)
            self.k_effective_history.append(k_eff)
            
            generation_data.append({
                'generation': generation,
                'neutrons_start': current_neutrons,
                'neutrons_end': next_neutrons,
                'fissions': fissions,
                'absorptions': absorptions,
                'leakages': leakages,
                'k_effective': k_eff
            })
            
            print(f"{generation:3d} | {current_neutrons:8d} | {fissions:8d} | {absorptions:11d} | {leakages:8d} | {k_eff:5.3f}")
            
            current_neutrons = next_neutrons
            generation += 1
        
        # Calculate final statistics
        total_interactions = self.total_fissions + self.total_absorptions + self.total_leakages
        avg_k_eff = np.mean(self.k_effective_history) if self.k_effective_history else 0
        
        results = {
            'generation_data': generation_data,
            'total_generations': generation,
            'total_fissions': self.total_fissions,
            'total_absorptions': self.total_absorptions,
            'total_leakages': self.total_leakages,
            'total_interactions': total_interactions,
            'average_k_effective': avg_k_eff,
            'final_neutrons': current_neutrons,
            'generation_sizes': self.generation_sizes,
            'k_effective_history': self.k_effective_history
        }
        
        print(f"\nSimulation completed after {generation} generations")
        print(f"Total fissions: {self.total_fissions}")
        print(f"Total absorptions: {self.total_absorptions}")
        print(f"Total leakages: {self.total_leakages}")
        print(f"Average k-effective: {avg_k_eff:.3f}")
        
        return results
    
    def plot_results(self, results):
        """Plot simulation results"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        generations = range(len(results['generation_sizes']))
        
        # Plot 1: Neutron population vs generation
        ax1.semilogy(generations, results['generation_sizes'], 'b-o', markersize=4)
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Number of Neutrons')
        ax1.set_title('Neutron Population vs Generation')
        ax1.grid(True)
        
        # Plot 2: k-effective vs generation
        if results['k_effective_history']:
            ax2.plot(generations, results['k_effective_history'], 'r-o', markersize=4)
            ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.7, label='Critical (k=1)')
            ax2.set_xlabel('Generation')
            ax2.set_ylabel('k-effective')
            ax2.set_title('Multiplication Factor vs Generation')
            ax2.grid(True)
            ax2.legend()
        
        # Plot 3: Cumulative events
        generations_full = range(1, len(results['generation_data']) + 1)
        cum_fissions = np.cumsum([g['fissions'] for g in results['generation_data']])
        cum_absorptions = np.cumsum([g['absorptions'] for g in results['generation_data']])
        cum_leakages = np.cumsum([g['leakages'] for g in results['generation_data']])
        
        ax3.plot(generations_full, cum_fissions, 'g-', label='Fissions')
        ax3.plot(generations_full, cum_absorptions, 'r-', label='Absorptions')
        ax3.plot(generations_full, cum_leakages, 'b-', label='Leakages')
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Cumulative Count')
        ax3.set_title('Cumulative Neutron Interactions')
        ax3.legend()
        ax3.grid(True)
        
        # Plot 4: Event distribution per generation
        fissions_per_gen = [g['fissions'] for g in results['generation_data']]
        absorptions_per_gen = [g['absorptions'] for g in results['generation_data']]
        leakages_per_gen = [g['leakages'] for g in results['generation_data']]
        
        width = 0.25
        x = np.arange(len(generations_full))
        
        ax4.bar(x - width, fissions_per_gen, width, label='Fissions', color='green', alpha=0.7)
        ax4.bar(x, absorptions_per_gen, width, label='Absorptions', color='red', alpha=0.7)
        ax4.bar(x + width, leakages_per_gen, width, label='Leakages', color='blue', alpha=0.7)
        
        ax4.set_xlabel('Generation')
        ax4.set_ylabel('Number of Events')
        ax4.set_title('Neutron Interactions per Generation')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def run_multiple_simulations(n_runs=100, **kwargs):
    """Run multiple simulations to get statistics"""
    print(f"Running {n_runs} Monte Carlo simulations...")
    
    k_eff_values = []
    final_generations = []
    
    simulator = NeutronChainReaction()
    
    for run in range(n_runs):
        if (run + 1) % 20 == 0:
            print(f"Completed {run + 1}/{n_runs} runs")
        
        # Run simulation quietly
        import sys
        from io import StringIO
        
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        results = simulator.run_simulation(**kwargs)
        
        sys.stdout = old_stdout
        
        k_eff_values.append(results['average_k_effective'])
        final_generations.append(results['total_generations'])
    
    print(f"\nResults from {n_runs} simulations:")
    print(f"Average k-effective: {np.mean(k_eff_values):.3f} ± {np.std(k_eff_values):.3f}")
    print(f"Average generations: {np.mean(final_generations):.1f} ± {np.std(final_generations):.1f}")
    
    # Plot distribution of k-effective values
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(k_eff_values, bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(np.mean(k_eff_values), color='red', linestyle='--', 
                label=f'Mean = {np.mean(k_eff_values):.3f}')
    plt.axvline(1.0, color='green', linestyle='--', label='Critical = 1.0')
    plt.xlabel('k-effective')
    plt.ylabel('Frequency')
    plt.title('Distribution of k-effective Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(final_generations, bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(np.mean(final_generations), color='red', linestyle='--',
                label=f'Mean = {np.mean(final_generations):.1f}')
    plt.xlabel('Number of Generations')
    plt.ylabel('Frequency')
    plt.title('Distribution of Simulation Lengths')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return k_eff_values, final_generations

# Example usage
if __name__ == "__main__":
    # Create simulator with parameters that give k ≈ 1.75 (supercritical)
    simulator = NeutronChainReaction(
        fission_probability=0.85,      # High fission probability
        absorption_probability=0.10,   # Low absorption
        leakage_probability=0.05,      # Low leakage
        neutrons_per_fission_mean=2.4, # Average neutrons per fission
        neutrons_per_fission_std=0.3   # Some variation
    )
    
    print("=== Single Detailed Simulation ===")
    results = simulator.run_simulation(initial_neutrons=1, max_generations=15)
    simulator.plot_results(results)
    
    print("\n=== Multiple Simulation Statistics ===")
    k_eff_dist, gen_dist = run_multiple_simulations(
        n_runs=50, 
        initial_neutrons=1, 
        max_generations=20,
        min_neutrons=1
    )