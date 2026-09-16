import secrets
import time

P3072 = int(
    "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74"
    "020BBEA63B139B22514A08798E3404DDEF9519B3CD3A431B302B0A6DF25F1437"
    "4FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7ED"
    "EE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3DC2007CB8A163BF05"
    "98DA48361C55D39A69163FA8FD24CF5F83655D23DCA3AD961C62F356208552BB"
    "9ED529077096966D670C354E4ABC9804F1746C08CA18217C32905E462E36CE3B"
    "E39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF695581718"
    "3995497CEA956AE515D2261898FA051015728E5A8AAAC42DAD33170D04507A33"
    "A85521ABDF1CBA64ECFB850458DBEF0A8AEA71575D060C7DB3970F85A6E1E4C7"
    "ABF5AE8CDB0933D71E8C94E04A25619DCEE3D2261AD2EE6BF12FFA06D98A0864"
    "D87602733EC86A64521F2B18177B200CBBE117577A615D6C770988C0BAD946E2"
    "08E24FA074E5AB3143DB5BFCE0FD108E4B82D120A93AD2CAFFFFFFFFFFFFFFFF", 16)
Q3072 = (P3072 - 1) // 2
G3072 = 2


EXP_BITS = 256


def modexp_cost(samples=40):
    xs = [secrets.randbits(EXP_BITS) for _ in range(samples)]
    t0 = time.perf_counter()
    for x in xs:
        pow(G3072, x, P3072)
    return (time.perf_counter() - t0) / samples


class ElGamal:
    def __init__(self):
        self.p, self.q, self.g = P3072, Q3072, G3072
        self.sk = secrets.randbits(EXP_BITS)
        self.pk = pow(self.g, self.sk, self.p)

    def enc(self, m):
        r = secrets.randbits(EXP_BITS)
        return (pow(self.g, r, self.p), m * pow(self.pk, r, self.p) % self.p)

    def mul(self, c1, c2):
        return (c1[0] * c2[0] % self.p, c1[1] * c2[1] % self.p)

    def exp(self, c, k):
        return (pow(c[0], k, self.p), pow(c[1], k, self.p))

    def dec(self, c):
        return c[1] * pow(pow(c[0], self.sk, self.p), -1, self.p) % self.p

    def pet(self, c1, c2):
        inv = (pow(c2[0], -1, self.p), pow(c2[1], -1, self.p))
        d = self.mul(c1, inv)
        z = secrets.randbits(EXP_BITS) + 1
        return self.dec(self.exp(d, z)) == 1


def jcj_cleansing_ops(n_ballots, n_voters):
    """Pairwise plaintext equality tests, as in JCJ and Civitas."""
    dup = n_ballots * (n_ballots - 1) // 2
    roster = n_ballots * n_voters
    return {"pets": dup + roster, "exps": 10 * (dup + roster)}


def chide_cleansing_ops(n_ballots, n_voters, lam=128, n_trustees=3):
    """Sorting-network based cleansing with bitwise encrypted credentials."""
    n = n_ballots + n_voters
    if n < 2:
        return {"gates": 0, "exps": 0}
    levels = max(1, (n - 1).bit_length())
    comparators = n * levels * (levels + 1) // 4
    bits = lam + max(1, (n_ballots - 1).bit_length()) + 1
    gates = comparators * bits
    gates += n * (2 * bits + 1)
    return {"gates": gates, "exps": 4 * n_trustees * gates}
