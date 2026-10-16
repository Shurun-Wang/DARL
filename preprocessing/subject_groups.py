"""Verified duplicate recordings: retain IDs, but split them as one unit."""

MDD_DUPLICATE_GROUPS = (("H26", "H29"), ("P32", "P33"))


def subject_group(dataset, subject):
    if dataset == "mdd":
        for members in MDD_DUPLICATE_GROUPS:
            if subject in members:
                return members[0]
    return subject
