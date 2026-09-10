#!/usr/bin/env python3
"""
Bootstrap script for Phase 1 GraphKnowledge.

Runs the complete pipeline:
1. Chunk Meditations
2. Extract entities + relationships
3. Deduplicate
4. Build knowledge graph
5. Test queries
"""

import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase1_pipeline import run_phase1

# Meditations Books I-VIII (you provided this)
MEDITATIONS_TEXT = """BOOK I.

THE EMPEROR MARCUS AURELIUS ANTONINUS:

HIS MEDITATIONS;

OR, DISCOURSES WITH HIMSELF.

BOOK I.

THE example of my grandfather Verus gave me a good disposition, not prone to anger.

2. By the recollection of my father's character, I learned to be both modest
and manly.

3. As for my mother, she taught me to have regard for religion, to be generous and open-handed, and not only to forbear from doing anybody an ill-turn, but not so much as to endure the thought of it. By her likewise I was bred to a plain, inexpensive way of living, very different from the common luxury of the rich.

4. I have to thank my great-grandfather that I did not go to a public school, but had good masters at home, and learnt to know that one ought to spend liberally on such things.

5. From my governor I learned not to join either the green or the blue faction on the race-ground, nor to support the Parmularius or Scutarius at the gladiators' shows. He taught me also to put my own hand to business upon occasion, to endure hardship and fatigues, and to throw the necessities of nature into a little compass; that I ought not to meddle with other people's business, nor be easy in giving credit to informers.

6. From Diognetus, to shun vain pursuits, not to be led away with the impostures of wizards and soothsayers, who pretend they can discharge evil spirits, and do strange feats by the strength of a charm; not to keep quails for the pit, nor to be eager after any such thing. This Diognetus taught me to bear freedom and plain-dealing in others, and apply myself to philosophy. He also procured me the instruction of Bacchius, Tandasias, and Marcianus. He likewise put me upon improving myself by writing dialogues when I was a boy; prevailed with me to prefer a couch covered with hides to a bed of state; and reconciled me to other like rigours of the Grecian discipline.

7. It was Rusticus that first made me desire to live rightly, and come to a better state; who prevented me from running into the vanity of the sophists, either by writing speculative treatises, haranguing upon moral subjects, or making a fantastical appearance or display of generosity or discipline. This philosopher kept me from yielding to the charms of rhetoric and poetry, from affecting the character of a man of pleasantry, from wearing my senator's robe in the house, or anything of this kind which looks like conceit and affectation. He taught me to write letters in a plain, unornamental style, like that dated by him from Sinuessa to my mother. By his instructions I was persuaded to be easily reconciled to those who had misbehaved themselves and disobliged me, as soon as they desired reconciliation. And of the same master I learned to read an author carefully. Not to take up with a superficial view, or assent quickly to idle talkers. And, to conclude with him, he gave me his own copy of Epictetus's memoirs.

8. Apollonius taught me to give my mind its due freedom, and disengage it from dependence upon chance, and not to regard, though ever so little, anything uncountenanced by reason. To maintain an equality of temper, even in acute pains, and loss of children, or tedious sickness. His practice was an excellent instance, that a man may be forcible and yet unbend his humour as occasion requires. The heaviness and impertinence of his scholars could seldom rouse his ill-temper. As for his learning, and the peculiar happiness of his manner in teaching, he was so far from being proud of himself upon this score, that one might easily perceive, he thought it one of the least things which belonged to him. This great man let me into the true secret of receiving an obligation, without either lessening myself, or seeming ungrateful to my friend."""


def main():
    """Run Phase 1 pipeline."""
    output_dir = "data/graph"
    
    print("=" * 70)
    print("GraphKnowledge Phase 1 — Bootstrap")
    print("=" * 70)
    
    try:
        kg, passages, query_engine = run_phase1(MEDITATIONS_TEXT, output_dir)
        
        print("\n" + "=" * 70)
        print("PHASE 1 COMPLETE")
        print("=" * 70)
        print(f"\nArtifacts saved to {output_dir}/")
        print("\nNext: Run interactive queries")
        print("  query_engine.query_entity('virtue', max_hops=2)")
        print("  query_engine.query_entity('fear', max_hops=2)")
        print("  query_engine.query_entity('Marcus', max_hops=2)")
        
        # Optionally start interactive session
        response = input("\nStart interactive session? (y/n): ").strip().lower()
        if response == 'y':
            query_engine.interactive_session()
            
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()