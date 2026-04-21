import json
import random
import math
from collections import Counter, defaultdict
import os

# ============================================================
# 1. CHARGEMENT DES DONNÉES
# ============================================================

def load_passwords(filepath):
    """Charge les mots de passe depuis un fichier"""
    passwords = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            pwd = line.strip().rstrip('\\')
            if pwd and len(pwd) >= 2:
                passwords.append(pwd)
    return passwords

# ============================================================
# 2. MODÈLE MARKOV
# ============================================================

START = '\x00'  
END = '\x01'    

class MarkovChain:
    """Chaîne de Markov d'ordre N pour génération de mots de passe"""
    
    def __init__(self, order=2):
        self.order = order
        self.transitions = defaultdict(Counter)
        self.total_counts = defaultdict(int)
        
    def train(self, passwords):
        """Entraîne le modèle sur les mots de passe"""
        for pwd in passwords:
            # Padding avec caractères spéciaux single-char
            padded = (START * self.order) + pwd + END
            
            for i in range(len(padded) - self.order):
                state = padded[i:i + self.order]
                next_char = padded[i + self.order]
                self.transitions[state][next_char] += 1
                self.total_counts[state] += 1
                
    def generate_one(self, max_length=20):
        """Génère un seul mot de passe"""
        state = START * self.order
        password = []
        
        for _ in range(max_length):
            if state not in self.transitions:
                break
                
            chars = self.transitions[state]
            total = self.total_counts[state]
            
            # Sélection pondérée
            rand = random.randint(1, total)
            cumul = 0
            chosen = None
            for char, count in chars.items():
                cumul += count
                if cumul >= rand:
                    chosen = char
                    break
            
            if chosen is None or chosen == END:
                break
                
            password.append(chosen)
            state = state[1:] + chosen
            
        return ''.join(password)
    
    def generate(self, n=10000, max_length=20):
        """Génère n mots de passe"""
        generated = set()
        attempts = 0
        max_attempts = n * 10
        
        while len(generated) < n and attempts < max_attempts:
            pwd = self.generate_one(max_length)
            if len(pwd) >= 2:
                generated.add(pwd)
            attempts += 1
            
        return generated
    
    def log_probability(self, password):
        """Calcule la log-probabilité d'un mot de passe"""
        padded = (START * self.order) + password + END
        log_prob = 0.0
        
        for i in range(len(padded) - self.order):
            state = padded[i:i + self.order]
            next_char = padded[i + self.order]
            
            if state not in self.transitions or next_char not in self.transitions[state]:
                return float('-inf')
            
            prob = self.transitions[state][next_char] / self.total_counts[state]
            log_prob += math.log(prob)
            
        return log_prob
    
    def get_stats(self):
        """Statistiques du modèle"""
        return {
            'order': self.order,
            'unique_states': len(self.transitions),
            'total_transitions': sum(self.total_counts.values()),
            'avg_choices_per_state': sum(len(v) for v in self.transitions.values()) / max(len(self.transitions), 1)
        }

# ============================================================
# 3. ANALYSE ET ÉVALUATION
# ============================================================

def evaluate_coverage(generated, eval_passwords):
    """Évalue combien de mots de passe eval sont dans les générés"""
    eval_set = set(eval_passwords)
    matches = generated & eval_set
    coverage = len(matches) / len(eval_set) * 100 if eval_set else 0
    return {
        'eval_total': len(eval_set),
        'matches': len(matches),
        'coverage_pct': round(coverage, 4),
        'matched_examples': list(matches)[:20]
    }

def analyze_generated(generated, train_passwords):
    """Analyse la qualité des mots de passe générés"""
    gen_list = list(generated)
    train_set = set(train_passwords)
    
    # Distribution des longueurs
    lengths = [len(p) for p in gen_list]
    
    # Composition
    has_lower = sum(1 for p in gen_list if any(c.islower() for c in p))
    has_upper = sum(1 for p in gen_list if any(c.isupper() for c in p))
    has_digit = sum(1 for p in gen_list if any(c.isdigit() for c in p))
    has_special = sum(1 for p in gen_list if any(not c.isalnum() for c in p))
    
    # Overlap avec train
    overlap = generated & train_set
    
    total = len(gen_list)
    return {
        'total_generated': total,
        'avg_length': round(sum(lengths) / max(total, 1), 2),
        'min_length': min(lengths) if lengths else 0,
        'max_length': max(lengths) if lengths else 0,
        'length_distribution': dict(Counter(lengths).most_common(15)),
        'pct_with_lower': round(has_lower / max(total, 1) * 100, 2),
        'pct_with_upper': round(has_upper / max(total, 1) * 100, 2),
        'pct_with_digit': round(has_digit / max(total, 1) * 100, 2),
        'pct_with_special': round(has_special / max(total, 1) * 100, 2),
        'overlap_with_train': len(overlap),
        'overlap_pct': round(len(overlap) / max(total, 1) * 100, 2),
        'overlap_examples': list(overlap)[:20]
    }

def compute_perplexity(model, passwords, sample_size=1000):
    """Calcule la perplexité du modèle sur un échantillon"""
    sample = random.sample(passwords, min(sample_size, len(passwords)))
    total_log_prob = 0
    total_chars = 0
    valid = 0
    
    for pwd in sample:
        lp = model.log_probability(pwd)
        if lp > float('-inf'):
            total_log_prob += lp
            total_chars += len(pwd) + 1  # +1 pour END token
            valid += 1
    
    if valid == 0 or total_chars == 0:
        return float('inf'), 0
        
    avg_log_prob = total_log_prob / total_chars
    perplexity = math.exp(-avg_log_prob)
    return round(perplexity, 4), valid

# ============================================================
# 4. MAIN
# ============================================================

def main():
    random.seed(42)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train_path = os.path.join(base_dir, 'data', 'splits', 'train.txt')
    eval_path = os.path.join(base_dir, 'data', 'splits', 'eval.txt')
    output_path = os.path.join(base_dir, 'output', 'results', 'markov_results.json')
    
    print("=" * 60)
    print("CHAÎNES DE MARKOV - GÉNÉRATION DE MOTS DE PASSE")
    print("=" * 60)
    
    # Chargement
    print("\n📂 Chargement des données...")
    train_pwds = load_passwords(train_path)
    eval_pwds = load_passwords(eval_path)
    print(f"   Train: {len(train_pwds)} mots de passe")
    print(f"   Eval:  {len(eval_pwds)} mots de passe")
    
    results = {}
    
    # Entraîner et tester les ordres 1 à 4
    orders = [1, 2, 3, 4]
    gen_per_order = 500_000  # Même volume que le Transformer pour comparaison équitable
    
    for order in orders:
        print(f"\n{'='*60}")
        print(f"🔗 MARKOV ORDRE {order}")
        print(f"{'='*60}")
        
        # Entraînement
        print(f"   Entraînement...")
        model = MarkovChain(order=order)
        model.train(train_pwds)
        stats = model.get_stats()
        print(f"   États uniques: {stats['unique_states']}")
        print(f"   Transitions totales: {stats['total_transitions']}")
        print(f"   Choix moyen/état: {stats['avg_choices_per_state']:.1f}")
        
        # Perplexité sur train (échantillon)
        print(f"   Calcul de perplexité (train)...")
        perp_train, valid_train = compute_perplexity(model, train_pwds, sample_size=2000)
        print(f"   Perplexité train: {perp_train} (sur {valid_train} valides)")
        
        # Perplexité sur eval
        print(f"   Calcul de perplexité (eval)...")
        perp_eval, valid_eval = compute_perplexity(model, eval_pwds, sample_size=len(eval_pwds))
        print(f"   Perplexité eval: {perp_eval} (sur {valid_eval} valides)")
        
        # Génération
        print(f"   Génération de {gen_per_order} mots de passe...")
        generated = model.generate(n=gen_per_order, max_length=20)
        print(f"   Générés uniques: {len(generated)}")
        
        # Exemples
        examples = random.sample(list(generated), min(30, len(generated)))
        print(f"\n   📋 30 exemples générés:")
        for i in range(0, min(30, len(examples)), 5):
            row = examples[i:i+5]
            print(f"      {' | '.join(row)}")
        
        # Évaluation coverage
        print(f"\n   📊 Coverage sur eval...")
        coverage = evaluate_coverage(generated, eval_pwds)
        print(f"   Matches: {coverage['matches']}/{coverage['eval_total']} ({coverage['coverage_pct']}%)")
        if coverage['matched_examples']:
            print(f"   Exemples matchés: {coverage['matched_examples'][:10]}")
        
        # Analyse qualité
        print(f"\n   📈 Analyse qualité des générés...")
        quality = analyze_generated(generated, train_pwds)
        print(f"   Longueur moyenne: {quality['avg_length']}")
        print(f"   Avec minuscules: {quality['pct_with_lower']}%")
        print(f"   Avec chiffres: {quality['pct_with_digit']}%")
        print(f"   Avec majuscules: {quality['pct_with_upper']}%")
        print(f"   Avec spéciaux: {quality['pct_with_special']}%")
        print(f"   Overlap avec train: {quality['overlap_with_train']} ({quality['overlap_pct']}%)")
        
        # Stocker résultats
        results[f'order_{order}'] = {
            'model_stats': stats,
            'perplexity_train': perp_train,
            'perplexity_train_valid': valid_train,
            'perplexity_eval': perp_eval,
            'perplexity_eval_valid': valid_eval,
            'generation': {
                'total_unique': len(generated),
                'examples': examples[:20]
            },
            'coverage': coverage,
            'quality': quality
        }
    
    # ============================================================
    # RÉSUMÉ COMPARATIF
    # ============================================================
    print(f"\n{'='*60}")
    print("📊 RÉSUMÉ COMPARATIF")
    print(f"{'='*60}")
    
    print(f"\n{'Ordre':<8} {'États':<10} {'Perp.Train':<12} {'Perp.Eval':<12} {'Générés':<10} {'Coverage':<10} {'Overlap%':<10}")
    print("-" * 72)
    for order in orders:
        r = results[f'order_{order}']
        print(f"{order:<8} {r['model_stats']['unique_states']:<10} {r['perplexity_train']:<12} {r['perplexity_eval']:<12} {r['generation']['total_unique']:<10} {r['coverage']['coverage_pct']:<10} {r['quality']['overlap_pct']:<10}")
    
    # Meilleur modèle par coverage
    best_order = max(orders, key=lambda o: results[f'order_{o}']['coverage']['coverage_pct'])
    print(f"\n🏆 Meilleur coverage: Ordre {best_order} ({results[f'order_{best_order}']['coverage']['coverage_pct']}%)")
    
    # Meilleur modèle par perplexité eval
    best_perp = min(orders, key=lambda o: results[f'order_{o}']['perplexity_eval'] if results[f'order_{o}']['perplexity_eval'] != float('inf') else 9999999)
    print(f"🏆 Meilleure perplexité eval: Ordre {best_perp} ({results[f'order_{best_perp}']['perplexity_eval']})")
    
    # Sauvegarder
    results['summary'] = {
        'best_coverage_order': best_order,
        'best_perplexity_order': best_perp,
        'generation_count_per_model': gen_per_order
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n💾 Résultats sauvegardés dans {output_path}")

if __name__ == '__main__':
    main()
