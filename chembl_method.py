async def fetch_chembl_ligands(self, uniprot_id: str, gene_name: str) -> dict:
    """
    Fetch ChEMBL ligands/compounds associated with a protein target.
    Returns top ligands with bioactivity data.
    """
    import requests
    cache_key = f"chembl_{uniprot_id}"
    cached = self.cache.get(cache_key)
    if cached:
        return cached
    
    try:
        # Step 1: Search for target by UniProt ID or gene name
        chembl_url = "https://www.ebi.ac.uk/chembl/api/data/target"
        search_params = {
            'format': 'json',
            'limit': 1,
            'organism': 'Homo sapiens'
        }
        
        # Try searching by UniProt ID first
        target_id = None
        search_params['target_synonym__icontains'] = uniprot_id
        response = requests.get(chembl_url, params=search_params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('results'):
                target_id = data['results'][0].get('target_chembl_id')
        
        # If not found, try gene name
        if not target_id:
            search_params['target_synonym__icontains'] = gene_name
            response = requests.get(chembl_url, params=search_params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('results'):
                    target_id = data['results'][0].get('target_chembl_id')
        
        if not target_id:
            return {"available": False, "ligands": [], "gene_name": gene_name}
        
        # Step 2: Get activities/compounds for this target
        activities_url = f"https://www.ebi.ac.uk/chembl/api/data/activity"
        activity_params = {
            'format': 'json',
            'limit': 10,
            'target_chembl_id': target_id,
            'type': 'IC50,EC50,Ki,Kd',
        }
        
        response = requests.get(activities_url, params=activity_params, timeout=10)
        if response.status_code != 200:
            return {"available": False, "ligands": [], "gene_name": gene_name}
        
        data = response.json()
        ligands = []
        
        for activity in data.get('results', [])[:10]:
            compound_chembl_id = activity.get('molecule_chembl_id')
            if not compound_chembl_id:
                continue
            
            try:
                compound_url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{compound_chembl_id}"
                compound_response = requests.get(compound_url, params={'format': 'json'}, timeout=10)
                if compound_response.status_code == 200:
                    compound_data = compound_response.json()
                    ligands.append({
                        'name': activity.get('ligand_name', 'Unknown'),
                        'chembl_id': compound_chembl_id,
                        'canonical_smiles': compound_data.get('canonical_smiles', ''),
                        'mw': compound_data.get('molecular_weight'),
                        'logp': compound_data.get('alogp'),
                        'hbd': compound_data.get('num_h_donors'),
                        'hba': compound_data.get('num_h_acceptors'),
                        'activity_type': activity.get('type'),
                        'activity_value': activity.get('value'),
                        'activity_unit': activity.get('units'),
                        'chembl_url': f"https://www.ebi.ac.uk/chembl/compound_report_card/{compound_chembl_id}/"
                    })
            except Exception:
                continue
        
        result = {
            "available": len(ligands) > 0,
            "target_id": target_id,
            "ligands": ligands,
            "gene_name": gene_name
        }
        
        self.cache.set(cache_key, result)
        return result
        
    except Exception as e:
        st.warning(f"ChEMBL API error: {str(e)}")
        return {"available": False, "ligands": [], "gene_name": gene_name}
