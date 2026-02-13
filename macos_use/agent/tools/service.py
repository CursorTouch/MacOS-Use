from macos_use.agent.tools.views import Click, Type, Scroll, Move, Shortcut, Wait, Scrape, Done, Shell, Memory, App, MultiSelect, MultiEdit, Desktop
from macos_use.agent.desktop.service import Desktop as _Desktop
from typing import Literal,Optional
from macos_use.tool import Tool
from pathlib import Path
from time import sleep

memory_path=Path.cwd()/'.memories'
memory_path.mkdir(parents=True, exist_ok=True)

def _resolve_memory_path(path: str) -> Path:
    """Resolve a memory file path, always sandboxed within .memories directory."""
    file_path = (memory_path / path).resolve()
    if not str(file_path).startswith(str(memory_path.resolve())):
        raise ValueError(f"Path escapes the .memories directory: {path}")
    return file_path

@Tool('done_tool',model=Done)
def done_tool(answer:str,**kwargs):
    '''
    Delivers a response to the user. This is the ONLY way to communicate with the user.

    MUST be called for every type of response:
    - Task completion: summarize what was accomplished
    - Answers to questions: provide the requested information
    - Conversational replies: greetings, clarifications, explanations
    - Error reports: explain what failed and why

    The answer should be formatted in clean markdown.
    '''
    return answer

@Tool('app_tool',model=App)
def app_tool(mode:Literal['launch','resize','switch']='launch',name:Optional[str]=None,loc:Optional[list[int]]=None,size:Optional[list[int]]=None,**kwargs)->str:
    '''
    Manages application windows: launch new apps, switch between open windows, or resize/reposition the active window.

    - launch: Opens an application via the Start Menu. Provide the app name as it appears in Start Menu.
    - switch: Brings an already-open window to the foreground. Provide the window title from the Window Info list.
    - resize: Moves and resizes the currently active window. Provide loc=[x,y] for position and size=[w,h] for dimensions.
    '''
    desktop:_Desktop=kwargs['desktop']
    return desktop.app(mode,name,loc,size)

@Tool('memory_tool',model=Memory)
def memory_tool(mode: Literal['view','read','write','delete','update'],path: Optional[str] = None,
    content: Optional[str] = None,operation: Optional[Literal['replace', 'insert']] = 'replace',
    old_str: Optional[str] = None,new_str: Optional[str] = None,line_number: Optional[int] = None,
    read_range: Optional[list[int]] = None,**kwargs) -> str:
    '''
    Persistent file-based storage for saving and retrieving information across steps. Files are stored as markdown in the .memories directory.

    - view: List all stored memory files.
    - write: Create a new file. Requires path and content.
    - read: Retrieve file contents. Optionally use read_range=[start, end] for partial reads.
    - update: Modify an existing file. Use operation='replace' with old_str/new_str, or operation='insert' with line_number/content.
    - delete: Remove a file by path.

    Use for storing intermediate results, research findings, plans, or any data needed in later steps.
    '''
    match mode:
        case 'view':
            search_dir = (memory_path / path).resolve() if path else memory_path
            if not str(search_dir).startswith(str(memory_path.resolve())):
                return "Error: path escapes the .memories directory."
            if not search_dir.exists():
                return "No memory files found."
            files = search_dir.rglob('*.md')
            result = '\n'.join([f'{i+1}. {file.relative_to(memory_path).as_posix()}' 
                               for i, file in enumerate(files)])
            return result if result else "No memory files found."
        
        case 'write':
            if not path:
                return "Error: path is required for write mode."
            if not path.endswith('.md'):
                path += '.md'
            file_path = _resolve_memory_path(path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding='utf-8')
            return f'{file_path.name} created in {file_path.parent.relative_to(memory_path).as_posix()}.'
        
        case 'read':
            if not path:
                return "Error: path is required for read mode."
            file_path = _resolve_memory_path(path)
            if not file_path.exists():
                return f'Error: {file_path.name} not found.'
            
            file_content = file_path.read_text(encoding='utf-8')
            
            if read_range:
                start, end = read_range
                lines = file_content.splitlines()
                
                if start < 0 or start >= len(lines):
                    return f'Error: start line {start} out of range (0-{len(lines)-1}).'
                if end < start or end > len(lines):
                    return f'Error: end line {end} out of range ({start}-{len(lines)}).'
                
                selected_lines = lines[start:end]
                return f"File: {file_path.relative_to(memory_path).as_posix()}\nLines {start}-{end-1}:\n" + '\n'.join(selected_lines)
            
            return f"File: {file_path.relative_to(memory_path).as_posix()}\nContent:\n{file_content}"
        
        case 'update':
            if not path:
                return "Error: path is required for update mode."
            file_path = _resolve_memory_path(path)
            if not file_path.exists():
                return f'Error: {file_path.name} not found. Use "write" mode to create a new file.'
            
            current_content = file_path.read_text(encoding='utf-8')
            
            match operation:
                case 'replace':
                    if not old_str or not new_str:
                        return 'Error: both old_str and new_str are required for replace operation.'
                    if old_str not in current_content:
                        return f'Error: "{old_str}" not found in file.'
                    
                    new_content = current_content.replace(old_str, new_str)
                    file_path.write_text(new_content, encoding='utf-8')
                    old_display = f'"{old_str[:50]}{"..." if len(old_str) > 50 else ""}"'
                    new_display = f'"{new_str[:50]}{"..." if len(new_str) > 50 else ""}"'
                    return f'{file_path.name} updated: replaced {old_display} with {new_display}.'
                
                case 'insert':
                    if line_number is None:
                        return 'Error: line_number is required for insert operation.'
                    if not content:
                        return 'Error: content is required for insert operation.'
                    
                    lines = current_content.splitlines(keepends=True)
                    if line_number < 0 or line_number > len(lines):
                        return f'Error: line_number {line_number} out of range (0-{len(lines)}).'
                    
                    lines.insert(line_number, content + '\n' if not content.endswith('\n') else content)
                    new_content = ''.join(lines)
                    file_path.write_text(new_content, encoding='utf-8')
                    return f'{file_path.name} updated: inserted content at line {line_number}.'
                
                case _:
                    return f'Error: Unknown operation "{operation}".'
        
        case 'delete':
            if not path:
                return "Error: path is required for delete mode."
            file_path = _resolve_memory_path(path)
            if not file_path.exists():
                return f'Error: {file_path.name} not found.'
            
            file_path.unlink()
            return f'{file_path.name} deleted from {file_path.parent.relative_to(memory_path).as_posix()}.'
        
    return "Invalid mode. Use 'view', 'read', 'write', 'update', or 'delete'."

@Tool('shell_tool',model=Shell)
def shell_tool(command: str,mode:Literal['shell','osascript']='shell',timeout:int=10,**kwargs) -> str:
    '''
    Executes a command and returns the output and exit status code.

    - shell: Runs a bash command in Terminal. Working directory is the user's HOME. Use for file operations, system queries, installations, running scripts, and any task better done via command line than GUI.
    - osascript: Executes AppleScript code. Use for advanced macOS automation such as controlling apps, manipulating windows, interacting with System Events, displaying dialogs, or querying app-specific properties.

    Check the status code in the response: 0 means success, non-zero means failure.
    '''
    desktop:_Desktop=kwargs['desktop']
    response,status=desktop.execute_command(command,mode=mode,timeout=timeout)
    return f'Response: {response}\nStatus Code: {status}'

@Tool('click_tool',model=Click)
def click_tool(loc:Optional[list[int]]=None,button:Literal['left','right','middle']='left',clicks:int=1,**kwargs)->str:
    '''
    Clicks at the specified pixel coordinates on screen.

    - Single left click (clicks=1): Select elements, press buttons, focus fields, follow links.
    - Double left click (clicks=2): Open files, folders, and desktop icons.
    - Right click (button='right'): Open context menus.
    - Hover only (clicks=0): Move cursor to location without clicking.

    Use coordinates from the Interactive Elements list in the Desktop State.
    '''
    x,y=loc
    desktop:_Desktop=kwargs['desktop']
    desktop.click(loc,button,clicks)
    num_clicks={1:'Single',2:'Double',3:'Triple'}
    return f'{num_clicks.get(clicks)} {button} clicked at ({x},{y}).'

@Tool('type_tool',model=Type)
def type_tool(loc:Optional[list[int]]=None,text:str='',clear:Literal['true','false']='false',caret_position:Literal['start','idle','end']='idle',press_enter:Literal['true','false']='false',**kwargs):
    '''
    Clicks an input field and types text into it. Do NOT pre-click with click_tool — this tool handles focusing automatically.

    - Set clear=true to replace existing text, or clear=false to append.
    - Set press_enter=true to submit after typing (search bars, forms, dialogs).
    - Set caret_position to control where typing begins relative to existing text.

    Use for search queries, form fields, text editors, address bars, and any text input.
    '''
    x,y=loc
    desktop:_Desktop=kwargs['desktop']
    desktop.type(loc=loc,text=text,caret_position=caret_position,clear=clear,press_enter=press_enter)
    return f'Typed {text} at ({x},{y}).'

@Tool('scroll_tool',model=Scroll)
def scroll_tool(loc:Optional[list[int]]=None,type:Literal['horizontal','vertical']='vertical',direction:Literal['up','down','left','right']='down',wheel_times:int=1,**kwargs)->str:
    '''
    Scrolls content at the specified location or at the current cursor position.

    Each wheel increment scrolls roughly 3-5 lines of text. Use wheel_times=3-5 for moderate scrolling, 10+ for large jumps.

    Check scroll percentages in the Scrollable Elements list to gauge position before scrolling. If loc is omitted, scrolling occurs at the current cursor location.
    '''
    desktop:_Desktop=kwargs['desktop']
    response=desktop.scroll(loc,type,direction,wheel_times)
    if response:
        return response
    return f'Scrolled {type} {direction} by {wheel_times} wheel times.'

@Tool('move_tool',model=Move)
def move_tool(loc:list[int],drag:bool=False,**kwargs)->str:
    '''
    Moves the mouse cursor to a target location, or drags from the current position to the target.

    - drag=false: Hover over elements to reveal tooltips, dropdown menus, or reposition the cursor.
    - drag=true: Hold left mouse button from the current cursor position and drag to the target coordinates. Use for drag-and-drop of files, window resizing, slider adjustment, or reordering items.
    '''
    x,y=loc
    desktop:_Desktop=kwargs['desktop']
    if drag:
        desktop.drag(loc)
        return f'Dragged the selected element to ({x},{y}).'
    else:
        desktop.move(loc)
        return f'Moved the mouse pointer to ({x},{y}).'

@Tool('shortcut_tool',model=Shortcut)
def shortcut_tool(shortcut:str,**kwargs)->str:
    '''
    Presses a keyboard shortcut. Use '+' to combine keys (e.g., 'ctrl+c').

    Common shortcuts: ctrl+c (copy), ctrl+v (paste), ctrl+z (undo), ctrl+s (save), ctrl+a (select all), ctrl+f (find), ctrl+w (close tab), ctrl+t (new tab), alt+tab (switch window), alt+f4 (close app), enter (confirm), escape (cancel), win (start menu).

    Prefer shortcuts over mouse interactions when the operation is unambiguous.
    '''
    desktop:_Desktop=kwargs['desktop']
    desktop.shortcut(shortcut)
    return f'Pressed {shortcut}.'

@Tool('multi_select_tool',model=MultiSelect)
def multi_select_tool(press_ctrl:Literal['true','false']='true',elements:list[list[int]]=[],**kwargs)->str:
    '''
    Clicks multiple locations in sequence. With press_ctrl=true, holds Ctrl to accumulate a multi-selection (e.g., selecting multiple files, checkboxes, or list items). With press_ctrl=false, clicks each location independently in order.

    Provide a list of [x, y] coordinates. Each is clicked once in the order given.
    '''
    desktop:_Desktop=kwargs['desktop']
    desktop.multi_select(press_ctrl,elements)
    elements_str = '\n'.join([f"({x},{y})" for x,y in elements])
    return f'Multi-selected elements at {elements_str}.'

@Tool('multi_edit_tool',model=MultiEdit)
def multi_edit_tool(elements:list[list],**kwargs)->str:
    '''
    Types text into multiple input fields in one action. Each entry is [x, y, text]: the tool clicks the location and types the text, then moves to the next entry.

    Use for filling out forms with multiple fields (name, email, address) or editing several text inputs at once. More efficient than calling type_tool repeatedly.
    '''
    desktop:_Desktop=kwargs['desktop']
    desktop.multi_edit(elements)
    elements_str = ','.join([f'({x},{y}) text={text}' for x,y,text in elements])
    return f'Multi-edited elements at {elements_str}.'

@Tool('wait_tool',model=Wait)
def wait_tool(duration:int,**kwargs)->str:
    '''
    Pauses for the specified number of seconds before the next action. Use to wait for applications to launch, pages to load, dialogs to appear, or animations to finish. Typical waits: 2-3s for UI transitions, 5s for app launches, 10s+ for installations or downloads.
    '''
    sleep(duration)
    return f'Waited for {duration} seconds.'

@Tool('desktop_tool', model=Desktop)
def desktop_tool(action: Literal['create', 'remove', 'rename', 'switch'], desktop_name: Optional[str] = None, new_name: Optional[str] = None, **kwargs) -> str:
    '''
    Manages macOS virtual desktops (Spaces) for workspace organization via Mission Control.

    - create: Creates a new Space by opening Mission Control and adding a desktop.
    - remove: Removes the current Space. Switch to the target Space first using switch, then remove it.
    - switch: Switches to a Space by number (e.g., desktop_name="2") or direction (desktop_name="left"/"right"/"next"/"previous").
      Switching by number requires the shortcut to be enabled in System Settings > Keyboard > Shortcuts > Mission Control.
    - rename: Not supported on macOS. Spaces are identified by number, not name.
    '''
    desktop: _Desktop = kwargs['desktop']
    try:
        return desktop.manage_spaces(action, desktop_name, new_name)
    except Exception as e:
        return f"Error executing desktop action '{action}': {str(e)}"

@Tool('scrape_tool',model=Scrape)
def scrape_tool(url:str,**kwargs)->str:
    '''
    Extracts visible text content from the webpage currently displayed in the browser and returns it as markdown.

    This reads the rendered page content via the accessibility tree — not the raw HTML. Provide the URL of the page currently open in the browser. The output includes scroll position indicators so you know if there is more content above or below.

    Use when you need to read, analyze, or extract information from a webpage.
    '''
    desktop:_Desktop=kwargs['desktop']
    desktop_state=desktop.desktop_state
    tree_state=desktop_state.tree_state
    if not tree_state.dom_node:
        content=desktop.scrape(url)
        return f'URL:{url}\nContent:\n{content}'
    dom_node=tree_state.dom_node
    vertical_scroll_percent=dom_node.vertical_scroll_percent
    content='\n'.join([node.text for node in tree_state.dom_informative_nodes])
    header_status = "Reached top" if vertical_scroll_percent <= 0 else "Scroll up to see more"
    footer_status = "Reached bottom" if vertical_scroll_percent >= 100 else "Scroll down to see more"
    return f'URL:{url}\nContent:\n{header_status}\n{content}\n{footer_status}'