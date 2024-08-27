
async def is_visible_in_viewport(page, locator):
    return await page.evaluate("""
    (element) => {
        function isVisibleInViewport(element) {
            const elementStyle = window.getComputedStyle(element);
            if (
                elementStyle.height == '0px' ||
                elementStyle.display == 'none' ||
                elementStyle.opacity == '0' ||
                elementStyle.visibility == 'hidden' ||
                elementStyle.clipPath == 'circle(0px at 50% 50%)' ||
                elementStyle.transform == 'scale(0)' ||
                element.hasAttribute('hidden')
            ) {
                return false;
            }

            const rect = element.getBoundingClientRect();
            
            const baseElementLeft = rect.left;
            const baseElementTop = rect.top;
            
            const elementFromStartingPoint = document.elementFromPoint(baseElementLeft, baseElementTop);
            
            if (elementFromStartingPoint != null && !element.isSameNode(elementFromStartingPoint)) {
                const elementZIndex = elementStyle.zIndex;
                const elementOverlappingZIndex = window.getComputedStyle(elementFromStartingPoint).zIndex;
                if (Number(elementZIndex) < Number(elementOverlappingZIndex)){
                    return false;
                }
                
                if (elementZIndex === '' && elementOverlappingZIndex === '') {
                    if (element.compareDocumentPosition(elementFromStartingPoint) & Node.DOCUMENT_POSITION_FOLLOWING) {
                        return false;
                    }
                }
            }
                
            return (
                rect.top >= 0 &&
                rect.left >= 0 &&
                rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
                rect.right <= (window.innerWidth || document.documentElement.clientWidth)
            );
        }
        return isVisibleInViewport(element);
    }
    """, locator.element_handle())