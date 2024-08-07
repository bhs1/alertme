const customCSS = `
    ::-webkit-scrollbar {
        width: 10px;
    }
    ::-webkit-scrollbar-track {
        background: #27272a;
    }
    ::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 0.375rem;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
`;

const styleTag = document.createElement("style");
styleTag.textContent = customCSS;
document.head.append(styleTag);

let labels = [];

function unmarkPage() {
  // Unmark page logic
  for (const label of labels) {
    document.body.removeChild(label);
  }
  labels = [];
}

function isTypable(element) {
    // Check if the element is inherently designed to accept text input
    const typableTags = ['TEXTAREA', 'INPUT'];
    const typableInputTypes = ['text', 'password', 'email', 'number', 'search', 'tel', 'url', 'date', 'datetime-local', 'time', 'week', 'month'];
    
    if (typableTags.includes(element.tagName)) {
        // Additional check for INPUT elements to filter by type
        if (element.tagName === 'INPUT') {
            return typableInputTypes.includes(element.type.toLowerCase());
        }
        return true; // TEXTAREA is always typable
    }
    
    // Additional check for contenteditable being true
    return element.contentEditable === 'true';
}

function isClickable(element) {
    // Check if the element is clickable
    return element.tagName === "A" ||
           element.onclick != null ||
           window.getComputedStyle(element).cursor == "pointer";
}

function isWithinScrollableArea(element) {
    let parent = element.parentElement;
    while (parent) {
      if (
        parent.scrollHeight > parent.clientHeight ||
        parent.scrollWidth > parent.clientWidth
      ) {
        const overflowY = window.getComputedStyle(parent).overflowY;
        const overflowX = window.getComputedStyle(parent).overflowX;
        if (
          overflowY === "auto" ||
          overflowY === "scroll" ||
          overflowX === "auto" ||
          overflowX === "scroll"
        ) {
          return true;
        }
      }
      parent = parent.parentElement;
    }
    return false;
}

function getParentHTML(element, maxLength = 500) {
  let currentElement = element;
  let previousHTML = currentElement.outerHTML;

  while (currentElement && currentElement.parentElement) {
    currentElement = currentElement.parentElement;
    if (currentElement.outerHTML.length > maxLength) {
      return previousHTML;
    }
    previousHTML = currentElement.outerHTML;
  }

  return previousHTML;
}

function markPage() {
  unmarkPage();

  var bodyRect = document.body.getBoundingClientRect();

  var items = Array.prototype.slice
    .call(document.querySelectorAll("*"))
    .map(function (element) {
      var vw = Math.max(
        document.documentElement.clientWidth || 0,
        window.innerWidth || 0
      );
      var vh = Math.max(
        document.documentElement.clientHeight || 0,
        window.innerHeight || 0
      );
      var textualContent = element.textContent.trim().replace(/\s{2,}/g, " ");
      var elementType = element.tagName.toLowerCase();
      var ariaLabel = element.getAttribute("aria-label") || "";

      var rects = [...element.getClientRects()]
        .filter((bb) => {
          var center_x = bb.left + bb.width / 2;
          var center_y = bb.top + bb.height / 2;
          var elAtCenter = document.elementFromPoint(center_x, center_y);

          return elAtCenter === element || element.contains(elAtCenter);
        })
        .map((bb) => {
          const rect = {
            left: Math.max(0, bb.left),
            top: Math.max(0, bb.top),
            right: Math.min(vw, bb.right),
            bottom: Math.min(vh, bb.bottom),
          };
          return {
            ...rect,
            width: rect.right - rect.left,
            height: rect.bottom - rect.top,
          };
        });

      var area = rects.reduce((acc, rect) => acc + rect.width * rect.height, 0);

      return {
        element: element,
        include:
          (element.tagName === "INPUT" ||
          element.tagName === "TEXTAREA" ||
          element.tagName === "SELECT" ||
          element.tagName === "BUTTON" ||
          element.tagName === "A" ||
          element.onclick != null ||
          window.getComputedStyle(element).cursor == "pointer" ||
          element.tagName === "IFRAME" ||
          element.tagName === "VIDEO") &&
          (isWithinScrollableArea(element) ||
          isClickable(element) ||
          isTypable(element)),
        area,
        rects,
        text: textualContent,
        type: elementType,
        ariaLabel: ariaLabel,
        isScrollableArea: isWithinScrollableArea(element),
        isTypeable: isTypable(element),
        isClickable: isClickable(element)
      };
    })
    .filter((item) => item.include && item.area >= 20);

  // Only keep inner clickable items
  items = items.filter(
    (x) => !items.some((y) => x.element.contains(y.element) && !(x == y))
  );

  // Function to generate random colors
  function getRandomColor() {
    var letters = "0123456789ABCDEF";
    var color = "#";
    for (var i = 0; i < 6; i++) {
      color += letters[Math.floor(Math.random() * 16)];
    }
    return color;
  }

  // Use a single loop to create labels for all items
  items.forEach(function (item, index) {
    const bbox = item.rects[0];

    newElement = document.createElement("div");
    var borderColor = getRandomColor();
    newElement.style.outline = `2px dashed ${borderColor}`;
    newElement.style.position = "fixed";
    newElement.style.left = bbox.left + "px";
    newElement.style.top = bbox.top + "px";
    newElement.style.width = bbox.width + "px";
    newElement.style.height = bbox.height + "px";
    newElement.style.pointerEvents = "none";
    newElement.style.boxSizing = "border-box";
    newElement.style.zIndex = 2147483647;

    // Add floating label at the bottom right corner
    var label = document.createElement("span");
    label.textContent = index;
    label.style.position = "absolute";
    label.style.bottom = "-18px";
    label.style.right = "-20px";
    label.style.background = borderColor;
    label.style.color = "white";
    label.style.padding = "2px 4px";
    label.style.fontSize = "12px";
    label.style.borderRadius = "2px";
    label.style.opacity = "0.5";
    newElement.appendChild(label);

    document.body.appendChild(newElement);
    labels.push(newElement);
  });

  function getXPathForElement(element) {
    const parts = [];
    while (element && element.nodeType === Node.ELEMENT_NODE) {
        let index = 0;
        let sibling = element.previousSibling;
        while (sibling) {
            if (sibling.nodeType === Node.DOCUMENT_TYPE_NODE) {
                sibling = sibling.previousSibling;
                continue;
            }
            if (sibling.nodeName === element.nodeName) {
                index++;
            }
            sibling = sibling.previousSibling;
        }
        const tagName = element.nodeName.toLowerCase();
        const part = index === 0 ? tagName : `${tagName}[${index + 1}]`;
        parts.unshift(part);
        element = element.parentNode;
    }
    return parts.length ? '/' + parts.join('/') : null;
  }

  const coordinates = items.flatMap((item, index) =>
    item.rects.map(({ left, top, width, height }) => ({
      x: left + (width / 2),
      y: top + (height / 2),
      type: item.type,
      text: item.text,
      ariaLabel: item.ariaLabel,
      outerHTML: item.element.outerHTML,
      parentHTML: getParentHTML(item.element),
      isScrollable: item.isScrollableArea,
      isTypeable: item.isTypeable,
      isClickable: item.isClickable,
      index: index,
      xpath: getXPathForElement(item.element)
    }))
  );

  return coordinates;
}