# Exercice 3 : pavage d'un rectangle avec des dominos
# ---------------------------------------------------
# On considère un rectangle de dimensions 3xN, et des dominos de
# dimensions 2x1. On souhaite calculer le nombre de façons de paver le
# rectangle avec des dominos.

# Ecrire une fonction qui calcule le nombre de façons de paver le
# rectangle de dimensions 3xN avec des dominos.
# Indice: trouver une relation de récurrence entre le nombre de façons
# de paver un rectangle de dimensions 3xN et le nombre de façons de
# paver un rectangle de dimensions 3x(N-1), 3x(N-2) et 3x(N-3).


def domino_paving(n: int) -> int:
    """
    Calcule le nombre de façons de paver un rectangle de dimensions 3xN
    avec des dominos.
    """
    a = 0
    # BEGIN SOLUTION
    if n == 0:
        return 1  
    elif n == 1:
        return 0 
    elif n == 2:
        return 3 
    elif n == 3:
        return 0  
    else:

        dp = [0] * (n + 1)
        dp[0] = 1
        dp[1] = 0
        dp[2] = 3
        dp[3] = 0

        for i in range(4, n + 1):
            a = 4 * dp[i-2] - dp[i-4] 
            dp[i] = a 
        
        return dp[n]
    # END SOLUTION
