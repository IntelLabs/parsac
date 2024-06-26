import torch


def coord_at_pos(rects, pos):
    if rects.ndim == 2:
        return rects[:, pos]
    return rects[pos].item()


def llx(rects):
    return coord_at_pos(rects, 0)


def lly(rects):
    return coord_at_pos(rects, 1)


def urx(rects):
    return coord_at_pos(rects, 2)


def ury(rects):
    return coord_at_pos(rects, 3)


def width(rects):
    return (urx(rects) - llx(rects))


def height(rects):
    return (ury(rects) - lly(rects))


def area(rects):
    return width(rects) * height(rects)


def is_overlap(test_rect, all_rects):
    overlapped_rects_x = torch.logical_and(
        llx(test_rect) < urx(all_rects), urx(test_rect) > llx(all_rects))
    overlapped_rects_y = torch.logical_and(
        lly(test_rect) < ury(all_rects), ury(test_rect) > lly(all_rects))

    return torch.logical_and(overlapped_rects_x, overlapped_rects_y)
