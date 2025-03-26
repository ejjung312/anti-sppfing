import cv2


def putTextRect(img, text, pos, scale=3, thickness=3, colorT=(255,255,255), colorR=(255,0,255),
                font=cv2.FONT_HERSHEY_PLAIN, offset=10, border=None, colorB=(0,255,0)):
    ox, oy = pos
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)

    # x1: 왼쪽 상단 x 좌표, y1: 왼쪽 상단 y 좌표, x2: 오른쪽 하단 x 좌표, y2: 오른쪽 하단 y 좌표
    x1, y1, x2, y2 = ox - offset, oy + offset, ox + w + offset, oy - h - offset

    cv2.rectangle(img, (x1,y1), (x2,y2), colorR, cv2.FILLED)
    if border is not None:
        cv2.rectangle(img, (x1,y1), (x2,y2), colorB, border)
    cv2.putText(img, text, (ox,oy), font ,scale, colorT, thickness)

    return img, [x1,y2,x2,y1]


def cornerRect(img, bbox, l=30, t=5, rt=1, colorR=(255,0,255), colorC=(0,255,0)):
    x,y,w,h = bbox
    x1,y1 = x+w, y+h
    if rt != 0:
        cv2.rectangle(img, bbox, colorR, rt)

    # Top Left x,y
    cv2.line(img, (x,y), (x+l, y), colorC, t)
    cv2.line(img, (x,y), (x, y+l), colorC, t)

    # Top Right x1,y
    cv2.line(img, (x1,y), (x1-l, y), colorC, t)
    cv2.line(img, (x1,y), (x1, y+l), colorC, t)

    # Bottom Left x,y1
    cv2.line(img, (x,y1), (x+l, y1), colorC, t)
    cv2.line(img, (x,y1), (x, y1-l), colorC, t)

    # Bottom Right x1,y1
    cv2.line(img, (x1,y1), (x1-l, y1), colorC, t)
    cv2.line(img, (x1,y1), (x1, y1-l), colorC, t)

    return img
















