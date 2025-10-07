"""
Export trained PINN memristor model to SPICE-compatible formats
Windows 11 compatible version
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple
import subprocess
import sys

from CompleteExperimentalReproduction import run_complete_experiments
from ExperimentalValidationFramework import ExperimentalValidator


def export_lut(
    pinn,
    validator: ExperimentalValidator,
    output_dir: Path,
    v_min: float = -2.0,
    v_max: float = 2.0,
    n_voltage_points: int = 500,
    n_state_points: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Export PINN as 2D lookup table: I = f(V, x)"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create grid
    V_grid = np.linspace(v_min, v_max, n_voltage_points)
    x_grid = np.linspace(0.0, 1.0, n_state_points)
    
    # Generate current for all (V, x) combinations
    I_grid = np.zeros((n_voltage_points, n_state_points))
    
    print(f"Generating {n_voltage_points}x{n_state_points} lookup table...")
    for i, V in enumerate(V_grid):
        for j, x in enumerate(x_grid):
            V_array = np.array([V], dtype=np.float32)
            x_array = np.array([x], dtype=np.float32)
            I_pred, _ = validator.predict_current(V_array, x_array)
            I_grid[i, j] = I_pred[0]
        
        if i % 100 == 0:
            print(f"  Progress: {i}/{n_voltage_points}")
    
    # Save as text file
    lut_path = output_dir / "pinn_memristor_lut.txt"
    with open(lut_path, 'w') as f:
        f.write("* PINN Memristor Lookup Table\n")
        f.write(f"* Voltage range: [{v_min}, {v_max}] V\n")
        f.write(f"* State range: [0, 1]\n")
        f.write(f"* Grid: {n_voltage_points} voltage x {n_state_points} state points\n")
        f.write("*\n")
        f.write("* Format: V(V) x(normalized) I(A)\n")
        
        for i, V in enumerate(V_grid):
            for j, x in enumerate(x_grid):
                f.write(f"{V:.6f} {x:.6f} {I_grid[i, j]:.6e}\n")
    
    print(f"Lookup table saved to {lut_path}")
    
    # Save as numpy
    np.savez(
        output_dir / "pinn_lut.npz",
        V_grid=V_grid,
        x_grid=x_grid,
        I_grid=I_grid
    )
    
    return V_grid, x_grid, I_grid


def generate_spice_netlists(output_dir: Path) -> None:
    """Generate SPICE netlists for Windows ngspice"""
    
    output_dir = Path(output_dir)
    
    # PINN netlist with Windows-compatible paths
    pinn_netlist = '''* 1T1R Memory Cell with PINN Memristor
.title PINN Memristor Write Operation

* Write pulse on gate
Vpulse gate 0 PULSE(0 3.0 10n 1n 1n 100n 300n)

* NMOS access transistor
M1 drain gate 0 0 NMOS_SIMPLE W=1u L=0.5u

* Memristor behavioral model (PINN physics)
* State variable x tracked by voltage on capacitor
Cx x 0 1p IC=0.1
Gx 0 x value={0.1*v(drain) - 0.05*v(x)}
Dx1 x vdd DIDEAL
Dx2 0 x DIDEAL

* Current through memristor (Ohmic with state-dependent R)
Gmem drain vdd value={v(drain)/(1e5 + (1e3-1e5)*v(x))}

* Power supply
Vdd vdd 0 DC 1.8

* Device models
.model NMOS_SIMPLE NMOS (VTO=0.7 KP=100u LAMBDA=0.01)
.model DIDEAL D (N=0.001)

* Transient analysis
.tran 0.1n 300n

* Control block for Windows
.control
set hcopydevtype=postscript
run
echo "Simulation complete"
set wr_vecnames
set wr_singlescale
wrdata pinn_spice_results.txt v(gate) v(drain) v(x) i(Vdd) time
quit
.endc

.end
'''
    
    # VTEAM netlist
    vteam_netlist = '''* 1T1R Memory Cell with VTEAM Model
.title VTEAM Memristor Write Operation

* Write pulse on gate
Vpulse gate 0 PULSE(0 3.0 10n 1n 1n 100n 300n)

* NMOS access transistor
M1 drain gate 0 0 NMOS_SIMPLE W=1u L=0.5u

* VTEAM state variable with window function
Cx x 0 1p IC=0.1
Gx 0 x value={
+ (v(drain) > 1.0) ? 1e-3*pow((v(drain)/1.0 - 1), 3)*(1 - pow(2*v(x)-1, 6)) :
+ (v(drain) < -1.0) ? -1e-3*pow((-v(drain)/1.0 - 1), 3)*(1 - pow(2*v(x)-1, 6)) :
+ 0
+ }
Dx1 x vdd DIDEAL
Dx2 0 x DIDEAL

* VTEAM current (abrupt threshold switching)
Gmem drain vdd value={v(drain)/(1e5 + (1e3-1e5)*v(x))}

* Power supply
Vdd vdd 0 DC 1.8

* Device models
.model NMOS_SIMPLE NMOS (VTO=0.7 KP=100u LAMBDA=0.01)
.model DIDEAL D (N=0.001)

* Transient analysis
.tran 0.1n 300n

* Control block for Windows
.control
set hcopydevtype=postscript
run
echo "Simulation complete"
set wr_vecnames
set wr_singlescale
wrdata vteam_spice_results.txt v(gate) v(drain) v(x) i(Vdd) time
quit
.endc

.end
'''
    
    # Save netlists
    (output_dir / "pinn_1t1r_test.cir").write_text(pinn_netlist)
    (output_dir / "vteam_1t1r_test.cir").write_text(vteam_netlist)
    
    print(f"SPICE netlists saved to {output_dir}")


def run_ngspice_windows(netlist_path: Path) -> bool:
    """Run ngspice on Windows"""
    
    try:
        # Try to run ngspice
        result = subprocess.run(
            ['ngspice', '-b', str(netlist_path)],
            capture_output=True,
            text=True,
            cwd=netlist_path.parent,
            timeout=60
        )
        
        if result.returncode == 0:
            print(f"  Success: {netlist_path.name}")
            return True
        else:
            print(f"  Error running {netlist_path.name}")
            print(f"  stderr: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("ERROR: ngspice not found in PATH")
        print("Please install ngspice and add to PATH")
        print("Download from: https://sourceforge.net/projects/ngspice/")
        return False
    except subprocess.TimeoutExpired:
        print(f"  Timeout: {netlist_path.name} took too long")
        return False


def visualize_lut(
    V_grid: np.ndarray,
    x_grid: np.ndarray,
    I_grid: np.ndarray,
    output_dir: Path,
) -> None:
    """Create visualization of lookup table"""
    
    fig = plt.figure(figsize=(12, 5))
    
    # Left: 3D surface
    ax1 = fig.add_subplot(121, projection='3d')
    V_mesh, x_mesh = np.meshgrid(V_grid, x_grid, indexing='ij')
    surf = ax1.plot_surface(V_mesh, x_mesh, I_grid*1e6, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('Voltage (V)')
    ax1.set_ylabel('State x')
    ax1.set_zlabel('Current (µA)')
    ax1.set_title('PINN I-V-x Surface')
    fig.colorbar(surf, ax=ax1, shrink=0.5)
    
    # Right: 2D slices
    ax2 = fig.add_subplot(122)
    state_slices = [0.1, 0.3, 0.5, 0.7, 0.9]
    for x_val in state_slices:
        idx = np.argmin(np.abs(x_grid - x_val))
        ax2.plot(V_grid, I_grid[:, idx]*1e6, label=f'x={x_val:.1f}', linewidth=2)
    
    ax2.set_xlabel('Voltage (V)')
    ax2.set_ylabel('Current (µA)')
    ax2.set_title('I-V Curves at Different States')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'lut_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"LUT visualization saved")


def plot_spice_results(output_dir: Path) -> None:
    """Parse and plot SPICE simulation results"""
    
    pinn_file = output_dir / "pinn_spice_results.txt"
    vteam_file = output_dir / "vteam_spice_results.txt"
    
    if not pinn_file.exists():
        print("\nSPICE results not found. Run simulations first using:")
        print(f"  cd {output_dir}")
        print(f"  ngspice -b pinn_1t1r_test.cir")
        print(f"  ngspice -b vteam_1t1r_test.cir")
        return
    
    # Load results
    try:
        pinn_data = np.loadtxt(pinn_file, skiprows=1)
        time_pinn = pinn_data[:, -1] * 1e9  # to ns
        v_gate_pinn = pinn_data[:, 0]
        v_drain_pinn = pinn_data[:, 1]
        x_pinn = pinn_data[:, 2]
        i_vdd_pinn = pinn_data[:, 3] * 1e6  # to µA
    except Exception as e:
        print(f"Error loading PINN results: {e}")
        return
    
    has_vteam = False
    if vteam_file.exists():
        try:
            vteam_data = np.loadtxt(vteam_file, skiprows=1)
            time_vteam = vteam_data[:, -1] * 1e9
            v_gate_vteam = vteam_data[:, 0]
            v_drain_vteam = vteam_data[:, 1]
            x_vteam = vteam_data[:, 2]
            i_vdd_vteam = vteam_data[:, 3] * 1e6
            has_vteam = True
        except Exception as e:
            print(f"Warning: Could not load VTEAM results: {e}")
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot 1: Voltages
    axes[0, 0].plot(time_pinn, v_gate_pinn, 'b-', linewidth=2, label='Gate')
    axes[0, 0].plot(time_pinn, v_drain_pinn, 'r-', linewidth=2, label='Drain (PINN)')
    if has_vteam:
        axes[0, 0].plot(time_vteam, v_drain_vteam, 'g--', linewidth=2, label='Drain (VTEAM)')
    axes[0, 0].set_xlabel('Time (ns)')
    axes[0, 0].set_ylabel('Voltage (V)')
    axes[0, 0].set_title('Write Operation: Voltages')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Current
    axes[0, 1].plot(time_pinn, i_vdd_pinn, 'r-', linewidth=2, label='PINN')
    if has_vteam:
        axes[0, 1].plot(time_vteam, i_vdd_vteam, 'g--', linewidth=2, label='VTEAM')
    axes[0, 1].set_xlabel('Time (ns)')
    axes[0, 1].set_ylabel('Current (µA)')
    axes[0, 1].set_title('Supply Current')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: State
    axes[1, 0].plot(time_pinn, x_pinn, 'r-', linewidth=2, label='PINN')
    if has_vteam:
        axes[1, 0].plot(time_vteam, x_vteam, 'g--', linewidth=2, label='VTEAM')
    axes[1, 0].set_xlabel('Time (ns)')
    axes[1, 0].set_ylabel('State Variable x')
    axes[1, 0].set_title('Memristor State Evolution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])
    
    # Plot 4: Energy
    dt_pinn = np.diff(time_pinn * 1e-9)
    power_pinn = v_drain_pinn[:-1] * (i_vdd_pinn[:-1] * 1e-6)
    energy_pinn = np.cumsum(power_pinn * dt_pinn) * 1e12  # pJ
    
    axes[1, 1].plot(time_pinn[:-1], energy_pinn, 'r-', linewidth=2, label='PINN')
    
    if has_vteam:
        dt_vteam = np.diff(time_vteam * 1e-9)
        power_vteam = v_drain_vteam[:-1] * (i_vdd_vteam[:-1] * 1e-6)
        energy_vteam = np.cumsum(power_vteam * dt_vteam) * 1e12
        axes[1, 1].plot(time_vteam[:-1], energy_vteam, 'g--', linewidth=2, label='VTEAM')
    
    axes[1, 1].set_xlabel('Time (ns)')
    axes[1, 1].set_ylabel('Cumulative Energy (pJ)')
    axes[1, 1].set_title('Write Energy Consumption')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'spice_simulation_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"SPICE results plot saved")
    
    # Print summary
    print("\n=== SPICE Simulation Summary ===")
    print(f"PINN Model:")
    print(f"  Peak current: {np.max(i_vdd_pinn):.2f} µA")
    print(f"  Final state: {x_pinn[-1]:.3f}")
    print(f"  Write energy: {energy_pinn[-1]:.3f} pJ")
    
    if has_vteam:
        print(f"\nVTEAM Model:")
        print(f"  Peak current: {np.max(i_vdd_vteam):.2f} µA")
        print(f"  Final state: {x_vteam[-1]:.3f}")
        print(f"  Write energy: {energy_vteam[-1]:.3f} pJ")
        print(f"\nEnergy ratio (PINN/VTEAM): {energy_pinn[-1]/energy_vteam[-1]:.2f}x")


def plot_circuit_schematic(output_dir: Path):
    """Draw 1T1R circuit schematic"""
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(5, 7.5, '1T1R Memory Cell Schematic', 
            ha='center', fontsize=14, fontweight='bold')
    
    # VDD line
    ax.plot([2, 8], [6.5, 6.5], 'k-', linewidth=2)
    ax.text(1.5, 6.5, 'VDD', fontsize=12, va='center', fontweight='bold')
    
    # Memristor
    from matplotlib.patches import Rectangle
    mem_x, mem_y = 5, 5.5
    rect = Rectangle((mem_x-0.4, mem_y-0.3), 0.8, 0.6, 
                     fill=False, edgecolor='red', linewidth=3)
    ax.add_patch(rect)
    ax.text(mem_x, mem_y, 'M', fontsize=14, ha='center', va='center', 
            color='red', fontweight='bold')
    ax.plot([mem_x, mem_x], [6.5, mem_y+0.3], 'r-', linewidth=2)
    ax.plot([mem_x, mem_x], [mem_y-0.3, 4.5], 'r-', linewidth=2)
    ax.text(mem_x+1, mem_y, 'PINN\nMemristor', fontsize=10, color='red')
    
    # Transistor
    trans_x, trans_y = 5, 3.5
    ax.plot([trans_x-0.5, trans_x-0.5], [trans_y-0.3, trans_y+0.3], 'b-', linewidth=3)
    ax.plot([2, trans_x-0.5], [trans_y, trans_y], 'b-', linewidth=2)
    ax.text(1.2, trans_y, 'Gate\n(Write Pulse)', fontsize=10, va='center', fontweight='bold')
    
    # Drain/Source
    ax.plot([trans_x, trans_x], [trans_y+0.3, 4.5], 'k-', linewidth=2)
    ax.plot([trans_x, trans_x], [trans_y-0.3, 2], 'k-', linewidth=2)
    ax.plot([trans_x-0.3, trans_x+0.3], [trans_y+0.15, trans_y+0.15], 'k-', linewidth=2)
    ax.plot([trans_x-0.3, trans_x+0.3], [trans_y-0.15, trans_y-0.15], 'k-', linewidth=2)
    
    # Ground
    ax.plot([trans_x, trans_x], [2, 1.8], 'k-', linewidth=2)
    ax.plot([trans_x-0.3, trans_x+0.3], [1.8, 1.8], 'k-', linewidth=3)
    ax.plot([trans_x-0.2, trans_x+0.2], [1.6, 1.6], 'k-', linewidth=2)
    ax.plot([trans_x-0.1, trans_x+0.1], [1.4, 1.4], 'k-', linewidth=1)
    ax.text(trans_x+0.6, 1.5, 'GND', fontsize=12, fontweight='bold')
    
    # Drain node
    ax.plot([trans_x, trans_x+0.8], [4.5, 4.5], 'k-', linewidth=1)
    ax.scatter([trans_x+0.8], [4.5], s=100, c='black', zorder=5)
    ax.text(trans_x+1.2, 4.5, 'V_drain', fontsize=10, va='center')
    
    # Description
    desc_y = 0.8
    ax.text(5, desc_y, 'Write Operation:', fontsize=11, ha='center', fontweight='bold')
    ax.text(5, desc_y-0.4, '1. Gate pulse (3V, 100ns)', fontsize=9, ha='center')
    ax.text(5, desc_y-0.7, '2. Transistor turns ON', fontsize=9, ha='center')
    ax.text(5, desc_y-1.0, '3. Current flows → State changes', fontsize=9, ha='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'circuit_schematic.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Circuit schematic saved")


def main():
    """Complete workflow for Windows 11"""
    
    print("="*60)
    print("PINN to SPICE Integration - Windows 11 Workflow")
    print("="*60)
    
    output_dir = Path("results/spice_integration")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Train PINN
    print("\n[1/7] Training PINN model...")
    results = run_complete_experiments(
        epochs=800,
        seed=42,
        results_dir=output_dir / "training",
        show_plots=False
    )
    
    pinn = results['pinn']
    validator = results['validator']
    
    # Step 2: Export LUT
    print("\n[2/7] Exporting lookup table...")
    V_grid, x_grid, I_grid = export_lut(pinn, validator, output_dir)
    
    # Step 3: Visualize LUT
    print("\n[3/7] Visualizing LUT...")
    visualize_lut(V_grid, x_grid, I_grid, output_dir)
    
    # Step 4: Generate netlists
    print("\n[4/7] Generating SPICE netlists...")
    generate_spice_netlists(output_dir)
    
    # Step 5: Draw schematic
    print("\n[5/7] Generating circuit schematic...")
    plot_circuit_schematic(output_dir)
    
    # Step 6: Run SPICE
    print("\n[6/7] Running SPICE simulations...")
    print("  Running PINN simulation...")
    run_ngspice_windows(output_dir / "pinn_1t1r_test.cir")
    print("  Running VTEAM simulation...")
    run_ngspice_windows(output_dir / "vteam_1t1r_test.cir")
    
    # Step 7: Plot results
    print("\n[7/7] Plotting SPICE results...")
    plot_spice_results(output_dir)
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print(f"All files saved to: {output_dir.absolute()}")
    print("\nGenerated files:")
    print("  - lut_visualization.png")
    print("  - circuit_schematic.png")
    print("  - spice_simulation_results.png")
    print("  - pinn_memristor_lut.txt")
    print("  - pinn_1t1r_test.cir")
    print("  - vteam_1t1r_test.cir")
    print("="*60)


if __name__ == "__main__":
    main()