# Linear Boltzmann Transport Equation (BTE), Adjoint Formulation & SciML Implementations

Αυτό το έγγραφο συνοψίζει τη μαθηματική διατύπωση της γραμμικής εξίσωσης μεταφοράς Boltzmann (BTE), τη συζυγή της μορφή (Adjoint), τη σύνδεσή της με αντίστροφα προβλήματα (Inverse Problems), καθώς και συγκεκριμένα παραδείγματα εφαρμογής με Physics-Informed Neural Networks (PINNs) και Monte Carlo εργαλεία (Geant4/GRAS).

---

## 1. Θεωρητικό Υπόβαθρο

### 1.1 Ευθύ Πρόβλημα (Forward Problem)
Η γραμμική εξίσωση μεταφοράς Boltzmann περιγράφει τη ροή σωματιδίων ακτινοβολίας στον φασικό χώρο $(\vec{r}, E, \hat{\Omega})$:

$$L \psi = S$$

όπου:
* $\psi(\vec{r}, E, \hat{\Omega})$: Η γωνιακή ροή (angular flux).
* $S(\vec{r}, E, \hat{\Omega})$: Η πηγή σωματιδίων.
* $L$: Ο τελεστής μεταφοράς:
  $$L\psi = \hat{\Omega} \cdot \nabla \psi + \Sigma_t(\vec{r}, E)\psi - \int_0^\infty dE' \int_{4\pi} d\Omega' \, \Sigma_s(\vec{r}, E' \to E, \hat{\Omega}' \to \hat{\Omega})\psi(\vec{r}, E', \hat{\Omega}')$$

### 1.2 Συζυγές Πρόβλημα (Adjoint Problem)
Ορίζεται μέσω της ιδιότητας του εσωτερικού γινομένου $\langle \psi^\dagger, L \psi \rangle = \langle L^\dagger \psi^\dagger, \psi \rangle$:

$$L^\dagger \psi^\dagger = S^\dagger$$

* **Φυσική Σημασία $\psi^\dagger$:** Αντιπροσωπεύει τη **σπουδαιότητα (importance)** — την πιθανότητα ένα σωματίδιο στη θέση $\vec{r}$ με ενέργεια $E$ και κατεύθυνση $\hat{\Omega}$ να καταλήξει σε έναν ευαίσθητο ανιχνευτή και να καταγράψει απόκριση:
  $$R = \langle S^\dagger, \psi \rangle = \langle \psi^\dagger, S \rangle$$
* **Αντιστροφή:** Στον τελεστή $L^\dagger$, η ροή αντιστρέφεται ($-\hat{\Omega} \cdot \nabla$) και οι ενεργές διατομές σκέδασης μεταπίπτουν από $E' \to E$ σε $E \to E'$. Επιτρέπει την ιχνηλάτηση ανάποδα (από τον ανιχνευτή προς την εξωτερική πηγή).

### 1.3 Αντίστροφο Πρόβλημα (Inverse Problem)
Αναζητούνται οι παράμετροι της πηγής $S$ ή οι ιδιότητες των υλικών $\Sigma$ από μετρήσεις του ανιχνευτή $D_{\text{meas}}$:

$$\min_{S \text{ ή } \Sigma} \frac{1}{2} \Vert{} \mathcal{M}(\psi) - D_{\text{meas}} \Vert{}^2 + \mathcal{R}(S)$$

Η συζυγής μέθοδος (**Adjoint State Method**) χρησιμοποιείται για τον άμεσο υπολογισμό των παραγώγων (gradients) του σφάλματος ως προς τις παραμέτρους χωρίς επαναληπτικές ευθείες προσομοιώσεις.

---

## 2. Μονοδιάστατη Υλοποίηση PINN (1D Slab Benchmark)

Έστω 1D μονοενεργειακή πλάκα πάχους $a$ με απορρόφηση και ισότροπη σκέδαση ($\mu = \cos\theta \in [-1, 1]$):

$$\mu \frac{\partial \psi(x, \mu)}{\partial x} + \Sigma_t \psi(x, \mu) = \frac{\Sigma_s}{2} \int_{-1}^1 \psi(x, \mu') d\mu' + S(x)$$

### 2.1 Συναρτήσεις Απώλειας (Loss Functions)

* **Forward Loss:**
  $$\mathcal{L}_{\text{forward}}(\theta) = \frac{1}{N} \sum_{i=1}^N \left[ \mu_i \frac{\partial \psi_\theta}{\partial x}\Big\vert{}_{(x_i, \mu_i)} + \Sigma_t \psi_\theta(x_i, \mu_i) - \frac{\Sigma_s}{2} \sum_j w_j \psi_\theta(x_i, \mu_j') - S(x_i) \right]^2 + \mathcal{L}_{\text{BC}}$$

* **Adjoint Loss (Reverse Transport):**
  $$\mathcal{L}_{\text{adjoint}}(\phi) = \frac{1}{N} \sum_{i=1}^N \left[ -\mu_i \frac{\partial \psi^\dagger_\phi}{\partial x}\Big\vert{}_{(x_i, \mu_i)} + \Sigma_t \psi^\dagger_\phi(x_i, \mu_i) - \frac{\Sigma_s}{2} \sum_j w_j \psi^\dagger_\phi(x_i, \mu_j') - S^\dagger(x_i) \right]^2$$

* **Source Inversion Loss:**
  $$\mathcal{L}(\theta, \vec{p}) = \mathcal{L}_{\text{PDE}}(\theta, \vec{p}) + \lambda \Vert{}\psi_\theta(a, \mu) - \psi_{\text{meas}}(\mu)\Vert{}^2$$

---

## 3. Παραδείγματα Κώδικα (PoC Snippets)

### 3.1 Python (DeepXDE) – Adjoint BTE Formulation
```python
import deepxde as dde
import torch
import numpy as np

sigma_t = 1.0
sigma_s = 0.7

# Gauss-Legendre quadrature points & weights για το ολοκλήρωμα στο mu [-1, 1]
quad_pts, quad_w = np.polynomial.legendre.leggauss(16)
quad_pts_t = torch.tensor(quad_pts, dtype=torch.float32)
quad_w_t = torch.tensor(quad_w, dtype=torch.float32)

def adjoint_bte(x, psi_dag):
    # x[:, 0]: Χωρική θέση x
    # x[:, 1]: Γωνιακό συνημίτονο mu = cos(theta)
    dpsi_dx = dde.grad.jacobian(psi_dag, x, i=0, j=0)
    
    # Σκέδαση (ολοκλήρωμα πάνω στο mu)
    # Για πλήρη υλοποίηση χρησιμοποιείται quadrature interpolation επί του δικτύου
    scatter_integral = 0.5 * sigma_s * torch.sum(psi_dag * quad_w_t)
    
    # Adjoint PDE: -mu * d(psi^dagger)/dx + sigma_t * psi^dagger = scatter + S_dagger
    return -x[:, 1:2] * dpsi_dx + sigma_t * psi_dag - scatter_integral
    
### 3.2     
    ### 3.2 Julia (NeuralPDE.jl) – Integro-Differential Formulation
Julia

using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimJL
import DomainSets: Interval

@parameters x mu
@variables psi_dag(..)
Ix = Integral(mu in Interval(-1.0, 1.0))
Dx = Differential(x)

# Adjoint όρος μεταφοράς: -mu * d(psi)/dx
eq = -mu * Dx(psi_dag(x, mu)) + Sigma_t * psi_dag(x, mu) ~ 
      (Sigma_s / 2) * Ix(psi_dag(x, mu)) + S_detector(x)

4. Βασικές Δημοσιεύσεις & Papers Αναφοράς
BTE, PINNs & Neural Transport

    Mishra, S., & Molinaro, R. (2021). Estimating parameters for radiative transfer problems using physics-informed neural networks.

        arXiv: https://arxiv.org/abs/2009.05262

    Lu, L., Meng, X., Mao, Z., & Karniadakis, G. E. (2021). DeepXDE: A deep learning library for solving differential equations. SIAM Review.

        DOI: 10.1137/19M1274067

        arXiv: https://arxiv.org/abs/1907.04502

    Kothari, K., et al. (2023). Solving the Linearized Boltzmann Transport Equation with Physics-Informed Neural Networks. Journal of Computational Physics.

        DOI: 10.1016/j.jcp.2023.112105

Adjoint Monte Carlo & Space Radiation Shielding

    Desorgher, L., et al. (2006). The Adjoint Monte Carlo Module in GEANT4. IEEE Transactions on Nuclear Science, 53(6), 3699-3704.

        DOI: 10.1109/TNS.2006.886071

    Santin, G., et al. (2005). GRAS: A general-purpose 3-D tool for space radiation environment effects analysis. IEEE Transactions on Nuclear Science, 52(6), 2294-2299.

        DOI: 10.1109/TNS.2005.860749

    Berger, M. J., & Seltzer, S. M. (1972). Calculation of Electron and Photon Flux Emerging from a Slab. NBS Technical Report.

        NASA ADS: 1972STIN...7313709B

5. Repositories στο GitHub & Τεκμηρίωση
Αποθετήρια Κώδικα (GitHub)

    DeepXDE Main Repository:

        https://github.com/lululxvi/deepxde

        Σχετικά αρχεία/παραδείγματα: Εξετάστε τον κατάλογο examples/pinn_forward/ και examples/pinn_inverse/ για integro-differential και radiative transfer scripts.

    NeuralPDE.jl (SciML):

        https://github.com/SciML/NeuralPDE.jl

        Σχετικά tests: test/neural_adapter_tests.jl και test/ide_tests.jl για IDEs και integro-differential operators.

    DiffEqFlux.jl (SciML):

        https://github.com/SciML/DiffEqFlux.jl

    Geant4 Source Code (Adjoint Process Module):

        https://github.com/Geant4/geant4/tree/master/source/processes/electromagnetic/adjoint

        Περιέχει την επίσημη C++ υλοποίηση για reverse boundary crossing, reverse tracks και adjoint biasing.

Documentation & Portals

    SciML Documentation Hub: https://docs.sciml.ai/

    NeuralPDE Documentation: https://docs.sciml.ai/NeuralPDE/stable/

    DeepXDE Documentation: https://deepxde.readthedocs.io/

    ESA Space Environment & GRAS Portal: https://space-env.esa.int/tools-buffer/gras/

    Geant4 Official Documentation: https://geant4.web.cern.ch/