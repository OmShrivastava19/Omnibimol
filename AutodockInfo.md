
## **Important Notes**

### **This Implementation Uses Simulated Docking Because:**

1. **Real AutoDock Vina requires:**
   - Backend server with Vina installed
   - Protein preparation (PDBQT format conversion)
   - Ligand preparation (PDBQT format)
   - Grid box configuration
   - Computational resources (CPU intensive)

2. **For Production AutoDock Vina:**
   - Set up Flask/FastAPI backend
   - Install: `conda install -c conda-forge autodock-vina`
   - Process: PDB → PDBQT (using Open Babel)
   - Run: `vina --receptor protein.pdbqt --ligand ligand.pdbqt --out results.pdbqt`
   - Return results to frontend

3. **Current simulation:**
   - Estimates binding based on known IC50/Ki values
   - Provides realistic-looking results for demo
   - Shows the full docking workflow UI

---

## **Testing**

Test with these drug targets:

1. **EGFR** (P00533) - Many known kinase inhibitors
2. **TP53** (P04637) - Some experimental compounds
3. **INS** (P01308) - Limited direct ligands

**Custom compounds to try:**
- Aspirin
- Ibuprofen
- Caffeine
- Metformin

The interface is now complete with ligand browsing, custom docking setup, and results visualization!